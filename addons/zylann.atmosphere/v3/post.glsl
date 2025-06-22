#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Rendered scene so far
layout(rgba16f, set = 0, binding = 0) uniform image2D u_color_image;
// Depth of the rendered scene so far
layout(binding = 1) uniform sampler2D u_depth_texture;

layout(binding = 2) uniform sampler2D u_cloud_render0;
layout(binding = 3) uniform sampler2D u_cloud_render1;

layout(push_constant, std430) uniform PcParams {
    vec2 raster_size; // 0..7
    vec2 cloud_raster_size; // 8...15
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

//

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void main() {
    ivec2 fragcoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(u_pc_params.raster_size);

    if (fragcoord.x >= size.x || fragcoord.y >= size.y) {
        return;
    }

    vec2 screen_uv = vec2(fragcoord) / vec2(size);

    vec4 color = imageLoad(u_color_image, fragcoord);

    vec4 cloud_data0 = texture_2d_bicubic(u_cloud_render0, screen_uv, u_pc_params.cloud_raster_size);
    vec2 cloud_data1 = texture_2d_bicubic(u_cloud_render1, screen_uv, u_pc_params.cloud_raster_size).rg;

    vec3 cloud_transmittance = cloud_data0.rgb;
    vec3 cloud_scattering = vec3(cloud_data0.a, cloud_data1);
    cloud_scattering = cloud_scattering * cloud_scattering;

    const float exposure = 1.0;
    color.rgb = color.rgb * cloud_transmittance + cloud_scattering * exposure;

    imageStore(u_color_image, fragcoord, color);
}
