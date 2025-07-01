#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Rendered scene so far
layout(rgba16f, set = 0, binding = 0) uniform image2D u_color_image;
// Depth of the rendered scene so far
layout(binding = 1) uniform sampler2D u_depth_texture;

layout(binding = 2) uniform sampler2D u_cloud_render0;
layout(binding = 3) uniform sampler2D u_cloud_render1;
layout(binding = 5) uniform sampler2D u_cloud_render3; // depth

// Camera
layout (binding = 6) uniform CamParams {
    mat4 inv_view_matrix;
    mat4 inv_projection_matrix;
} u_cam_params;

layout(push_constant, std430) uniform PcParams {
    vec2 raster_size; // 0..7
    vec2 cloud_raster_size; // 8...15
    float time;
    float debug;
    float depth_threshold;
    float bicubic;
} u_pc_params;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// `texture_2d_bicubic` from SunshineClouds2
//
// MIT License
//
// Copyright (c) 2025 David House
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

float w0(float a) {
    return (1.0/6.0)*(a*(a*(-a + 3.0) - 3.0) + 1.0);
}

float w1(float a)  {
    return (1.0/6.0)*(a*a*(3.0*a - 6.0) + 4.0);
}

float w2(float a) {
    return (1.0/6.0)*(a*(a*(-3.0*a + 3.0) + 3.0) + 1.0);
}

float w3(float a) {
    return (1.0/6.0)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
float g0(float a) {
    return w0(a) + w1(a);
}

float g1(float a) {
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
float h0(float a) {
    return -1.0 + w1(a) / (w0(a) + w1(a));
}

float h1(float a) {
    return 1.0 + w3(a) / (w2(a) + w3(a));
}

vec4 texture_2d_bicubic(sampler2D tex, vec2 uv, vec2 res) {
	uv = uv * res + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );

	float g0x = g0(fuv.x);
	float g1x = g1(fuv.x);
	float h0x = h0(fuv.x);
	float h1x = h1(fuv.x);
	float h0y = h0(fuv.y);
	float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - 0.5) / res;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - 0.5) / res;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - 0.5) / res;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - 0.5) / res;
	
    return g0(fuv.y) * (g0x * texture(tex, p0)  +
                        g1x * texture(tex, p1)) +
           g1(fuv.y) * (g0x * texture(tex, p2)  +
                        g1x * texture(tex, p3));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vec2 uv_to_ndc(vec2 uv) {
    // OpenGL
    // return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    // Vulkan
    return uv * vec2(2.0) + vec2(-1.0);
}

vec3 get_viewspace_position_from_screen_uv(vec2 screen_uv, float nonlinear_depth, mat4 inv_projection) {
    vec4 temp = inv_projection * vec4(uv_to_ndc(screen_uv), nonlinear_depth, 1.0);
    return temp.xyz / temp.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Depth-aware upscaling
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Derived from Heaps.io
// https://github.com/HeapsIO/heaps/blob/master/h3d/shader/DepthAwareUpsampling.hx
//
// The MIT License (MIT)
//
// Copyright (c) 2013 Nicolas Cannasse
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

void upsample_depth_aware(
    sampler2D low_data_tex0, 
    sampler2D low_data_tex1, 
    sampler2D low_depth_tex, 
    vec2 low_res, 
    sampler2D screen_depth_tex, 
    ivec2 screen_pixel,
    vec2 screen_res,
    mat4 inv_projection,
    float depth_threshold,
    bool bicubic,
    out vec4 result0,
    out vec4 result1,
    out vec4 debug
) {
    const vec2 offsets[9] = {
        vec2(-1.0, -1.0),
        vec2(0.0, -1.0),
        vec2(1.0, -1.0),

        vec2(-1.0, 0.0),
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),

        vec2(-1.0, 1.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0),
    };

    const float upres_factor = 2.0;
    vec2 low_pixel_center = floor(vec2(screen_pixel) / upres_factor) + vec2(0.5);

    vec2 screen_uv = vec2(screen_pixel) / screen_res;

    float screen_depth_nonlinear = texelFetch(screen_depth_tex, screen_pixel, 0).r;
    vec3 screen_depth_pos = get_viewspace_position_from_screen_uv(screen_uv, screen_depth_nonlinear, inv_projection);

    float edge = 0.0;
    
    vec2 nearest_low_uv = screen_uv;
    float best_diff = 2.0;

    debug = vec4(0.0);
    debug.x = screen_depth_pos.z;

    vec2 inv_low_res = vec2(1.0) / low_res;

    for (int i = 0; i < 9; ++i) {
        vec2 offset = offsets[i];

        vec2 nuv = (low_pixel_center + offset) * inv_low_res;

        float low_depth_nonlinear = texture(low_depth_tex, nuv).r;

        float diff = abs(low_depth_nonlinear - screen_depth_nonlinear);
        if (diff < best_diff) {
            nearest_low_uv = nuv;
            best_diff = diff;
        }

        vec3 low_depth_pos = get_viewspace_position_from_screen_uv(nuv, low_depth_nonlinear, inv_projection);
        float dist = abs(screen_depth_pos.z - low_depth_pos.z);
        debug.y = low_depth_pos.z;
        if (dist > depth_threshold) {
            edge = 1.0;
        }
    }

    if (edge == 1.0) {
        result0 = texture(low_data_tex0, nearest_low_uv);
        result1 = texture(low_data_tex1, nearest_low_uv);
        // result1.r = 1.0;
        // result1.g = 0.0;
    } else {
        if (bicubic) {
            result0 = texture_2d_bicubic(low_data_tex0, screen_uv, low_res);
            result1 = texture_2d_bicubic(low_data_tex1, screen_uv, low_res);
        } else {
            result0 = texture(low_data_tex0, screen_uv);
            result1 = texture(low_data_tex1, screen_uv);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void main() {
    ivec2 fragcoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(u_pc_params.raster_size);

    if (fragcoord.x >= size.x || fragcoord.y >= size.y) {
        return;
    }

    vec2 screen_ps = vec2(1.0) / vec2(size);
    vec2 screen_uv = vec2(fragcoord) * screen_ps;
    screen_uv += screen_ps * 0.5;

    vec4 color = imageLoad(u_color_image, fragcoord);

    // ivec2 cloud_coord = fragcoord / 2;
    ivec2 cloud_size = size / 2;
    vec2 cloud_ps = vec2(1.0) / vec2(cloud_size);
    // vec2 cloud_uv = vec2(cloud_coord) * cloud_ps;
    // cloud_uv += cloud_ps * 0.5;

    vec2 cloud_uv = screen_uv;

    vec4 cloud_data0;
    vec4 cloud_data1;
    vec4 upsample_debug;

    upsample_depth_aware(
        u_cloud_render0, 
        u_cloud_render1, 
        u_cloud_render3,
        vec2(cloud_size),
        u_depth_texture,
        fragcoord,
        size,
        u_cam_params.inv_projection_matrix,
        u_pc_params.depth_threshold,
        u_pc_params.bicubic != 0.0,
        cloud_data0,
        cloud_data1,
        upsample_debug
    );

    vec3 cloud_transmittance = vec3(
        cloud_data0.r,
        cloud_data0.g,
        cloud_data0.b
    );
    vec3 cloud_scattering = vec3(
        cloud_data0.a,
        cloud_data1.r,
        cloud_data1.g
    );

    cloud_scattering = cloud_scattering * cloud_scattering;

    const float exposure = 1.0;
    color.rgb = color.rgb * cloud_transmittance + cloud_scattering * exposure;

    // if (screen_uv.x < 0.4) {
    //     color.rgb = vec3(fract(upsample_debug.x));
    // }
    // if (screen_uv.x < 0.2) {
    //     color.rgb = vec3(fract(upsample_debug.y));
    // }

    imageStore(u_color_image, fragcoord, color);
}
