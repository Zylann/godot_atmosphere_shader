#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Rendered scene so far
layout(rgba16f, set = 0, binding = 0) uniform image2D u_color_image;
// Depth of the rendered scene so far
layout(binding = 1) uniform sampler2D u_depth_texture;

layout(binding = 2) uniform sampler2D u_cloud_render0;
layout(binding = 3) uniform sampler2D u_cloud_render1;
layout(binding = 4) uniform sampler2D u_cloud_render2;
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
    float reserved1;
    float reserved2;
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

// vec4 texture_2d_bicubic_boxblur(sampler2D tex, vec2 pos, vec2 res) {
//     vec2 s = vec2(1.0) / res;
//     vec4 c0 = texture_2d_bicubic(tex, pos + vec2(-s.x, 0.0), res);
//     vec4 c1 = texture_2d_bicubic(tex, pos + vec2(s.x, 0.0), res);
//     vec4 c2 = texture_2d_bicubic(tex, pos + vec2(0.0, -s.y), res);
//     vec4 c3 = texture_2d_bicubic(tex, pos + vec2(0.0, s.y), res);
//     return (c0 + c1 + c2 + c3) * 0.25;
// }

//

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float compute_linear_depth_from_nonlinear(vec2 screen_uv, float nonlinear_depth, out vec4 view_coords) {
	// Note, we assume Vulkan here, which has NDC in 0..1 (OpenGL is -1..1)
	vec3 ndc = vec3(screen_uv * 2.0 - 1.0, nonlinear_depth);
	view_coords = u_cam_params.inv_projection_matrix * vec4(ndc, 1.0);
	//view_coords.xyz /= view_coords.w;
	//float linear_depth = -view_coords.z; // Not what I want because it changes when looking around
    // TODO Could we avoid world coordinates?
	vec4 world_coords = u_cam_params.inv_view_matrix * view_coords;
	vec3 pos_world = world_coords.xyz / world_coords.w;
	vec3 cam_pos_world = (u_cam_params.inv_view_matrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	// I wonder if there is a faster way to get to that distance...
	float linear_depth = distance(cam_pos_world, pos_world);
    return linear_depth;
}

float compute_linear_depth(vec2 screen_uv, out vec4 view_coords) {
	float nonlinear_depth = texture(u_depth_texture, screen_uv).x;
	return compute_linear_depth_from_nonlinear(screen_uv, nonlinear_depth, view_coords);
}

void main() {
    ivec2 fragcoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(u_pc_params.raster_size);

    if (fragcoord.x >= size.x || fragcoord.y >= size.y) {
        return;
    }

    vec2 screen_ps = vec2(1.0) / vec2(size);
    vec2 screen_uv = vec2(fragcoord) * screen_ps;
    screen_uv += screen_ps * 0.5;
    screen_uv += screen_ps * u_pc_params.debug;

    vec4 view_coords;
    float linear_depth = compute_linear_depth(screen_uv, view_coords);

    vec4 color = imageLoad(u_color_image, fragcoord);

    // ivec2 cloud_coord = fragcoord / 2;
    ivec2 cloud_size = size / 2;
    vec2 cloud_ps = vec2(1.0) / vec2(cloud_size);
    // vec2 cloud_uv = vec2(cloud_coord) * cloud_ps;
    // cloud_uv += cloud_ps * 0.5;

    vec2 cloud_uv = screen_uv;

    // cloud_uv += cloud_ps * u_pc_params.debug;

    // vec4 cloud_data0 = texture(u_cloud_render0, cloud_uv);
    // vec4 cloud_data1 = texture(u_cloud_render1, cloud_uv);
    // vec4 cloud_data2 = texture(u_cloud_render2, cloud_uv);
    // float cloud_depth = texture(u_cloud_render3, cloud_uv).r;

    vec4 cloud_data0 = texture_2d_bicubic(u_cloud_render0, cloud_uv, u_pc_params.cloud_raster_size);
    vec4 cloud_data1 = texture_2d_bicubic(u_cloud_render1, cloud_uv, u_pc_params.cloud_raster_size);
    vec4 cloud_data2 = texture_2d_bicubic(u_cloud_render2, cloud_uv, u_pc_params.cloud_raster_size);
    float cloud_depth = texture_2d_bicubic(u_cloud_render3, cloud_uv, u_pc_params.cloud_raster_size).r;

    // cloud_data0.a = cloud_data0.a * cloud_data0.a;
    // cloud_data1 = cloud_data1 * cloud_data1;

    vec3 cloud_transmittance0 = vec3(
        cloud_data0.r,
        cloud_data0.g,
        cloud_data0.b
    );
    vec3 cloud_scattering0 = vec3(
        cloud_data0.a,
        cloud_data1.r,
        cloud_data1.g
    );
    vec3 cloud_transmittance1 = vec3(
        cloud_data1.b,
        cloud_data1.a,
        cloud_data2.r
    );
    vec3 cloud_scattering1 = vec3(
        cloud_data2.g,
        cloud_data2.b,
        cloud_data2.a
    );

    cloud_scattering0 = cloud_scattering0 * cloud_scattering0;
    cloud_scattering1 = cloud_scattering1 * cloud_scattering1;

    vec3 cloud_transmittance = cloud_transmittance0;
    vec3 cloud_scattering = cloud_scattering0;

    // Remove artifacts caused by low-res rendering:
    // - When rendering clouds and atmosphere, return multiple values in the low-res buffer:
    //     - A: integration done with the max depth among the screen depths covered by the low-res pixel
    //     - B: integration done with the min depth among the screen depths covered by the low-res pixel
    //          (This can be done during integration, store in a variable until we go past min depth) 
    //     - C: min(max depth, cloud depth)
    //          (we can estimate cloud depth as the first sample with density > 0.1)
    // - When compositing at full res, compare full-res depth with low-res result C.
    //     - If C is lower, clouds are in front, use B
    //     - If C is higher, clouds are behind, use A

    if (linear_depth < cloud_depth) {
        cloud_transmittance = cloud_transmittance1;
        cloud_scattering = cloud_scattering1;
    }
    // if (screen_uv.x < 0.5) {
        // if (fract(u_pc_params.time) > 0.25) {
        //     cloud_transmittance = cloud_transmittance1;
        //     cloud_scattering = cloud_scattering1;
        // } else {
        //     cloud_transmittance = cloud_transmittance0;
        //     cloud_scattering = cloud_scattering0;
        // }
    // }

    const float exposure = 1.0;
    color.rgb = color.rgb * cloud_transmittance + cloud_scattering * exposure;

    // if (screen_uv.x < 0.6) {
    //     color.rgb = vec3(fract(linear_depth * 0.01));
    // }
    // if (screen_uv.x < 0.4) {
    //     color.rgb = vec3(fract(cloud_depth * 0.01));
    // }
    // if (screen_uv.x < 0.2) {
    //     if (fract(u_pc_params.time) > 0.5) {
    //         color.rgb = cloud_transmittance1;
    //     } else {
    //         color.rgb = cloud_transmittance0;
    //     }
    // }

    imageStore(u_color_image, fragcoord, color);
}
