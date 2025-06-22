#[compute]
#version 450

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Based on SimonDevy's clouds project, with modifications
// https://github.com/simondevyoutube/Shaders_Clouds1/tree/main
//
// MIT License
//
// Copyright (c) 2022 simondevyoutube
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #define CLOUDS_RAYMARCHED_LIGHTING
#define CLOUDS_MAX_RAYMARCH_STEPS 24
#define CLOUDS_MAX_RAYMARCH_FSTEPS 4
#define CLOUDS_LIGHT_RAYMARCH_STEPS 6
#define CLOUDS_SECONDARY_LIGHT_RAYMARCH_STEPS 3

//#define FULL_RES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef FULL_RES
// Rendered scene so far
layout(rgba16f, set = 0, binding = 0) uniform image2D u_input_image;
#else
layout(rgba8, set = 0, binding = 0) uniform image2D u_output_image0;
layout(rg8, set = 0, binding = 1) uniform image2D u_output_image1;
#endif

// Depth of the rendered scene so far
layout(binding = 2) uniform sampler2D u_depth_texture;

// Grayscale cubemap weighting overall cloud density
layout(binding = 3) uniform samplerCube u_cloud_coverage_cubemap;
// Precomputed noise used to shape the clouds, tiling seamlessly
layout(binding = 4) uniform sampler3D u_cloud_shape_texture;
layout(binding = 5) uniform sampler3D u_cloud_detail_texture;
// Blue noise used for dithering
layout(binding = 6) uniform sampler2D u_blue_noise_texture;

// Parameters that don't change every frame
layout (binding = 7) uniform Params {
    mat4 world_to_model_matrix;
    
	float planet_radius;
    float atmosphere_height;

    float cloud_density_scale;// = 1.0;
	float cloud_light_density_scale;
	float cloud_light_reach;
    float cloud_bottom;// = 0.2; // In ratio of atmosphere height
    float cloud_top;// = 0.5; // In ratio of atmosphere height

	float cloud_coverage_factor;
	float cloud_coverage_bias;
	
    float cloud_shape_factor;// = 0.5;
    float cloud_shape_bias;// = 0.0;
    float cloud_shape_scale;// = 1.0;
	float cloud_shape_amount;// = 1.0;

    float cloud_detail_factor;// = 0.5;
    float cloud_detail_bias;// = 0.0;
    float cloud_detail_scale;// = 1.0;
	float cloud_detail_amount;// = 1.0;
	float cloud_detail_falloff_distance;

	// TODO Have a few more point lights?
	float point_light_pos_x;
	float point_light_pos_y;
	float point_light_pos_z;
	float point_light_radius;

	float night_light_energy;
	float reserved1;
} u_params;

// Camera
layout (binding = 8) uniform CamParams {
    mat4 inv_view_matrix;
    mat4 inv_projection_matrix;
} u_cam_params;

// Parameters that may change every frame
layout(push_constant, std430) uniform PcParams {
    vec2 raster_size; // 0..7
    float time; // 8..11
    float frame; // 12..15

    vec4 planet_center_viewspace; // 16..31 // w contains sphere_depth_factor

    vec4 sun_center_viewspace; // 32..47 // w is not used

    vec2 cloud_coverage_rotation_x; // 48..55
    float reserved1; // 56..59
    float reserved2; // 60..63
} u_pc_params;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const float PI = 3.14159265359;

// x = first hit, y = second hit. Equal if not hit.
vec2 ray_sphere(vec3 center, float radius, vec3 ray_origin, vec3 ray_dir) {
	vec3 oc = ray_origin - center;
	float b = dot( oc, ray_dir );
	vec3 qc = oc - b*ray_dir;
	float h = radius*radius - dot(qc, qc);
	if (h < 0.0) {
		// No intersection
		//return Vector2(-1.0, -1.0)
		return vec2(1000000.0, 1000000.0);
	}
	h = sqrt( h );
	return vec2( -b-h, -b+h );
}

float pow2(float x) {
    return x * x;
}

vec2 pow2_vec2(vec2 v) {
    return vec2(pow2(v.x), pow2(v.y));
}

float pow4(float x) {
    return x * x * x * x;
}

vec4 blend_colors(vec4 background, vec4 foreground) {
	float sa = 1.0 - foreground.a;
	float a = background.a * sa + foreground.a;
	if (a == 0.0) {
        // Note: originally I had it return `vec4(0.0)`, just like in Godot's Color::blend method.
        // If we are working with colors that would have been just fine.
        // But for some reason, the Godot renderer expects RGB channels of the input image to be preserved even when
        // they are fully transparent. When we blended stuff that was totally transparent we ended up getting black 
        // pixels.
		return background;
	} else {
		return vec4((background.rgb * background.a * sa + foreground.rgb * foreground.a) / a, a);
	}
}

vec2 vec2_rotate_90(vec2 v) {
	return vec2(-v.y, v.x);
}

float unlerp(float minv, float maxv, float v) {
	return (v - minv) / (maxv - minv);
}

float linearstep(float minv, float maxv, float v) {
	return clamp(unlerp(minv, maxv, v), 0.0, 1.0);
}

float remap(float v, float min0, float max0, float min1, float max1) {
	float t = unlerp(min0, max0, v);
	return mix(min1, max1, t);
}

float pand_p2(float x, float c, float w) {
	return clamp(1.0 - pow2(2.0 * (x - c) / w), 0.0, 1.0);
}

float band_p4(float x, float c, float w) {
	return clamp(1.0 - pow4(2.0 * (x - c) / w), 0.0, 1.0);
}

float band_p2s_unit(float x, float s) {
	return max(1.0 - pow2(max((abs(x) - 1.0 + s) / s, 0.0)), 0.0);
}

float band_p2s(float x, float w, float s) {
	return band_p2s_unit(x / w, s);
}

float length_sq_vec3(vec3 v) {
	return dot(v, v);
}

float max_vec3_component(vec3 v) {
	return max(v.x, max(v.y, v.z));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Clouds
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Settings carried around in cloud functions.
// We don't use uniforms directly to make the code a bit more portable.
struct CloudSettings {
	float ground_height;
	float bottom_height;
	float top_height;

	float density_scale;
	float light_density_scale;
	float light_reach;

    mat2 coverage_rotation;
    float coverage_factor;
    float coverage_bias;

	float shape_factor;
	float shape_bias;
    float shape_scale;
	float shape_amount;

	float detail_factor;
	float detail_bias;
	float detail_scale;
	float detail_amount;
	float detail_falloff_distance;
};

float get_planet_shadow(vec3 pos, float planet_radius, vec3 sun_dir) {
	float dp = clamp(dot(normalize(pos), sun_dir) * 4.0 + 0.5, 0.0, 1.0);
	return dp * dp;
	// return 1.0;
}

float henyey_greenstein(float g, float mu) {
	float gg = g * g;
	return (1.0 / (4.0 * PI))  * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

float dual_henyey_greenstein(float g, float costh) {
	const float dual_lobe_weight = 0.7;
	return mix(henyey_greenstein(-g, costh), henyey_greenstein(g, costh), dual_lobe_weight);
}

float phase_function(float g, float costh) {
	return dual_henyey_greenstein(g, costh);
}

struct Coverage {
	float combined;
	float map;
	float shell;
};

Coverage sample_sdf_low(vec3 pos, CloudSettings settings) {
	float distance_to_core = length(pos);
	// float height = distance_to_core - settings.bottom_height;
	// float height_ratio = height / (settings.top_height - settings.bottom_height);

	const float mid_height = 0.5 * (settings.top_height + settings.bottom_height);
	const float shell_height = settings.top_height - settings.bottom_height;
	float shell_sd = abs(distance_to_core - mid_height) - shell_height * 0.5;

	vec2 coverage_pos_2d = settings.coverage_rotation * pos.xz;
	vec3 coverage_uv = vec3(coverage_pos_2d.x, pos.y, coverage_pos_2d.y);
	float coverage = texture(u_cloud_coverage_cubemap, coverage_uv).r;
	float map_sd = (coverage * 2.0 - 1.0 + settings.coverage_bias) * distance_to_core;

	float sd = max(shell_sd, map_sd);
	
	return Coverage(sd, map_sd, shell_sd);
}

float sample_density(Coverage base, vec3 pos, vec3 cam_origin, float time, CloudSettings settings) {
	float base_sd = max(base.map * settings.coverage_factor, base.shell);
	float density = clamp(-base_sd, 0.0, 1.0);

	if (density <= 0.0) {
		return 0.0;
	}

	float distance_to_core = length(pos);
	float height_ratio = (distance_to_core - settings.bottom_height) / (settings.top_height - settings.bottom_height);
	float height_falloff = 0.5;

	density = band_p2s_unit(height_ratio * 2.0 - 1.0, 0.5) * density;

	vec3 shape_uv = pos * settings.shape_scale;
	float shape = texture(u_cloud_shape_texture, shape_uv).r;
	shape = (shape * settings.shape_factor + settings.shape_bias) * settings.shape_amount;
	shape = mix(shape, shape * height_ratio, height_falloff);

	float dc = distance(cam_origin, pos);
	float detail_falloff = max(1.0 - dc / settings.detail_falloff_distance, 0.0);
	float detail = 0.0;
	if (detail_falloff > 0.0) {
		vec3 detail_uv = pos * settings.detail_scale + vec3(time*0.02, 0.0, 0.0);
		detail = texture(u_cloud_detail_texture, detail_uv).r;
		detail = (detail * settings.detail_factor + settings.detail_bias) * settings.detail_amount;
		detail *= detail_falloff;
	}

	density = max(density - shape - detail, 0.0);

	return density * settings.density_scale;
}

const vec3 EXTINCTION_MULT = vec3(0.8, 0.8, 1.0);

// Adapted from: https://twitter.com/FewesW/status/1364629939568451587/photo/1
vec3 multi_octave_scatter(float density, float mu) {
	float attenuation = 0.2;
	float contribution = 0.2;
	float phase_attenuation = 0.5;

	float a = 1.0;
	float b = 1.0;
	float c = 1.0;
	float g = 0.85;
	const int scattering_octaves = 4;

	vec3 luminance = vec3(0.0);

	for (int i = 0; i < scattering_octaves; ++i) {
		float phase = phase_function(0.3 * c, mu);
		vec3 beers = exp(-density * EXTINCTION_MULT * a);

		luminance += b * phase * beers;

		a *= attenuation;
		b *= contribution;
		c *= (1.0 - phase_attenuation);
	}

	return luminance;
}

vec3 raymarch_light_energy(
	vec3 ray_origin, 
	vec3 ray_dir, 
	vec3 cam_origin, 
	vec3 cam_dir,
	float time, 
	float jitter,
	CloudSettings settings,
	int steps
) {
	float max_distance = (settings.top_height - settings.bottom_height) * settings.light_reach;

	float step_len = max_distance / float(steps);
	float total_density = 0.0;

	float dist_travelled = 0.0;
	// ray_origin += jitter * step_len;

	for (int i = 0; i < steps; ++i) {
		vec3 pos = ray_origin + ray_dir * dist_travelled;

		Coverage sd1 = sample_sdf_low(pos, settings);

		total_density += sample_density(sd1, pos, cam_origin, time, settings) * step_len;
		dist_travelled += step_len;
	}

	total_density *= settings.light_density_scale;

	const float mu = dot(cam_dir, ray_dir);

	vec3 beers_law = multi_octave_scatter(total_density, mu);
	vec3 powder = 1.0 - exp(-total_density * 2.0 * EXTINCTION_MULT);
	
	return beers_law * mix(2.0 * powder, vec3(1.0), mu * 0.5 + 0.5);
}

float calculate_point_light_energy_factor(vec3 pos, vec4 point_light) {
	return 1.1 * pow4(max(1.0 - distance(point_light.xyz, pos) / point_light.w, 0.0));
}

struct CloudResult {
	vec3 scattering;
	vec3 transmittance;
};

CloudResult default_cloud_result() {
	return CloudResult(vec3(0.0), vec3(1.0));
}

CloudResult raymarch_cloud(
	vec3 ray_origin, // in planet space
	vec3 ray_dir, 
	float t_begin, 
	float t_end, 
	float jitter,
	vec3 sun_dir, 
	float time, 
	vec3 cam_pos,
	CloudSettings settings,
	vec4 point_light,
	float night_light_energy
) {
	vec3 sun_color = vec3(1.0);
	const float cloud_light_multiplier = 50.0;
	vec3 sun_light = sun_color * cloud_light_multiplier;
	const float ambient_strength = 0.0;
	vec3 ambient = vec3(ambient_strength * sun_color);

	const float ray_dist = t_end - t_begin;
	// const float step_dropoff = linearstep(1.0, 0.0, pow4(dot(vec3(0.0, 1.0, 0.0), ray_dir)));

	const float lq_step_len = ray_dist / float(CLOUDS_MAX_RAYMARCH_STEPS);
	const float hq_step_len = lq_step_len / float(CLOUDS_MAX_RAYMARCH_FSTEPS);
	const float max_steps = CLOUDS_MAX_RAYMARCH_STEPS * CLOUDS_MAX_RAYMARCH_FSTEPS;

	const float offset = lq_step_len * jitter;
	float dist_travelled = t_begin;

	int hq_marcher_countdown = 0;

	// TODO Not used?
	// float previous_step_len = 0.0;

	CloudResult result = default_cloud_result();
	const float break_transmittance = 0.01;

	for (float i = 0.0; i < max_steps; ++i) {
		// TODO Is this really needed? We already do max steps which is the sum of all potential hq steps
		if (dist_travelled > t_end) {
			break;
		}

		vec3 pos = ray_origin + dist_travelled * ray_dir;
		Coverage sd1 = sample_sdf_low(pos, settings);

		float current_step_len = lq_step_len;//sd1;

		if (hq_marcher_countdown <= 0) {
			if (sd1.combined < hq_step_len) {
				// Hit some clouds, step back
				hq_marcher_countdown = CLOUDS_MAX_RAYMARCH_FSTEPS;
				dist_travelled += hq_step_len * jitter;
			} else {
				dist_travelled += current_step_len;
				continue;
			}
		}

		if (hq_marcher_countdown > 0) {
			--hq_marcher_countdown;

			if (sd1.combined < 0.0) {
				hq_marcher_countdown = CLOUDS_MAX_RAYMARCH_FSTEPS;

				float extinction = sample_density(sd1, pos, cam_pos, time, settings);

				if (extinction > 0.01) {
					vec3 light_energy = vec3(0.0);

					float planet_shadow = get_planet_shadow(pos, settings.ground_height, sun_dir);

					// Sun light
					if (planet_shadow > 0.0) {
						light_energy += planet_shadow * raymarch_light_energy(
							pos, 
							sun_dir, 
							cam_pos, 
							ray_dir, 
							time, 
							jitter, 
							settings, 
							CLOUDS_LIGHT_RAYMARCH_STEPS
						);
					}

					// Point lights
					// light_energy += calculate_point_light_energy(pos, point_light);
					// TODO Option to disable raymarching
					float pt_light_energy_f = calculate_point_light_energy_factor(pos, point_light);
					if (pt_light_energy_f > 0.0) {
						light_energy += vec3(0.6, 0.8, 1.0) * pt_light_energy_f * raymarch_light_energy(
							pos, 
							normalize(point_light.xyz - pos), 
							cam_pos, 
							ray_dir, 
							time, 
							jitter, 
							settings, 
							CLOUDS_SECONDARY_LIGHT_RAYMARCH_STEPS
						);
					}

					// Night light
					// TODO Option to use cheap height light?
					if (planet_shadow < 1.0) {
						light_energy += night_light_energy * raymarch_light_energy(
							pos, 
							normalize(pos), 
							cam_pos, 
							ray_dir, 
							time, 
							jitter, 
							settings,
							CLOUDS_SECONDARY_LIGHT_RAYMARCH_STEPS
						);
					}

					vec3 luminance = ambient + sun_light * light_energy;
					vec3 transmittance = exp(-extinction * hq_step_len * EXTINCTION_MULT);
					vec3 integ_scatt = luminance * (1.0 - transmittance);

					result.scattering += result.transmittance * integ_scatt;
					result.transmittance *= transmittance;

					if (max_vec3_component(result.transmittance) <= break_transmittance) {
						result.transmittance = vec3(0.0);
						break;
					}
				}
			}

			dist_travelled += hq_step_len;
		}

		// previous_step_len = current_step_len;
	}

	const vec3 cloud_color = vec3(1.0);
	result.scattering = result.scattering * cloud_color;
	result.transmittance = clamp(result.transmittance, vec3(0.0), vec3(1.0));
	return result;
}

CloudResult render_clouds(
	vec3 planet_center_viewspace,
	vec3 ray_origin,
	vec3 ray_dir,
	float linear_depth,
	mat4 inv_view_matrix,
    mat4 world_to_model_matrix,
	vec3 sun_dir_viewspace,
	float jitter,
    float time,
    CloudSettings cloud_settings,
	vec4 point_light,
	float night_light_energy
) {
	vec2 rs_clouds_top = ray_sphere(planet_center_viewspace, cloud_settings.top_height, ray_origin, ray_dir);

	CloudResult result = CloudResult(vec3(0.0), vec3(1.0));

	if (rs_clouds_top.x != rs_clouds_top.y) {
		vec2 rs_clouds_bottom = ray_sphere(planet_center_viewspace, cloud_settings.bottom_height, ray_origin, ray_dir);

		vec2 cloud_rs = rs_clouds_top;
		cloud_rs.x = max(cloud_rs.x, 0.0);
		cloud_rs.y = min(cloud_rs.y, linear_depth);

		if (cloud_rs.x < linear_depth
			// Don't compute clouds when opaque stuff occludes them,
			// when under the clouds layer.
			// This saves 0.5ms in ground views on a 1060
			&& (linear_depth > rs_clouds_bottom.y || rs_clouds_bottom.x > 0.0)
		) {
			mat4 view_to_model_matrix = world_to_model_matrix * inv_view_matrix;
			vec3 cam_pos_model = (view_to_model_matrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

			// When under the cloud layer, this improves quality significantly,
			// unfortunately entering the cloud layer causes a jarring transition
			if (length_sq_vec3(cam_pos_model) < pow2(cloud_settings.bottom_height)) {
				cloud_rs.x = rs_clouds_bottom.y;
			}

			vec3 ray_origin_model = (view_to_model_matrix * vec4(ray_origin, 1.0)).xyz;
			vec3 ray_dir_model = (view_to_model_matrix * vec4(ray_dir, 0.0)).xyz;
			vec3 sun_dir_model = (view_to_model_matrix * vec4(sun_dir_viewspace, 0.0)).xyz;

			CloudResult rr = raymarch_cloud(
				ray_origin_model, 
                ray_dir_model, 
                cloud_rs.x, 
                cloud_rs.y, 
                jitter, 
                sun_dir_model,
				time, 
				cam_pos_model,
                cloud_settings,
				point_light,
				night_light_energy
            );

			// const float exposure = 1.0;
			// inout_color.rgb = inout_color.rgb * rr.transmittance + rr.scattering * exposure;
			result = rr;
		}
	}

	return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float compute_linear_depth(vec2 screen_uv, out vec4 view_coords) {
	float nonlinear_depth = texture(u_depth_texture, screen_uv).x;
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

void main() {
    ivec2 fragcoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(u_pc_params.raster_size);

    if (fragcoord.x >= size.x || fragcoord.y >= size.y) {
        return;
    }

    vec2 screen_uv = vec2(fragcoord) / vec2(size);

    vec4 view_coords;
    float linear_depth = compute_linear_depth(screen_uv, view_coords);

	// We'll evaluate the atmosphere in view space
	vec3 ray_origin = vec3(0.0, 0.0, 0.0);
	vec3 ray_dir = normalize(view_coords.xyz - ray_origin);

	float atmosphere_radius = u_params.planet_radius + u_params.atmosphere_height;
	vec2 rs_atmo = ray_sphere(u_pc_params.planet_center_viewspace.xyz, atmosphere_radius, ray_origin, ray_dir);

	// TODO if we run this shader in a double-clip scenario,
	// we have to account for the near and far clips properly, so they can be composed seamlessly

	CloudResult cr = default_cloud_result();

	if (rs_atmo.x != rs_atmo.y) {
		float t_begin = max(rs_atmo.x, 0.0);
		float t_end = max(rs_atmo.y, 0.0);

		vec2 rs_ground = ray_sphere(
            u_pc_params.planet_center_viewspace.xyz, 
            u_params.planet_radius, 
            ray_origin, 
            ray_dir
        );
		float gd = 10000000.0;
		if (rs_ground.x != rs_ground.y) {
			gd = rs_ground.x;
		}
        float sphere_depth_factor = u_pc_params.planet_center_viewspace.w;
		linear_depth = mix(linear_depth, gd, sphere_depth_factor);

		t_end = min(t_end, linear_depth);

		vec3 sun_dir_viewspace = normalize(
            u_pc_params.sun_center_viewspace.xyz - u_pc_params.planet_center_viewspace.xyz
        );

		float time = u_pc_params.time;
		float frame = u_pc_params.frame;

		// Blue noise doesn't have low-frequency patterns, it looks less "noisy"
		// http://momentsingraphics.de/BlueNoise.html
		ivec2 blue_noise_uv = (fragcoord + ivec2(time * 100.0)) & ivec2(0xff);
		// ivec2 blue_noise_uv = fragcoord & ivec2(0xff);
		float jitter = texelFetch(u_blue_noise_texture, blue_noise_uv, 0).r;

		// Animating Noise For Integration Over Time
		// https://blog.demofox.org/2017/10/31/animating-noise-for-integration-over-time/		
		// const float golden_ratio = 1.61803398875;
		// jitter = fract(jitter + float(int(frame) % 32) * golden_ratio);

		// vec4 atmosphere = compute_atmosphere_v2(ray_origin, ray_dir, in_v_planet_center_viewspace,
		// 	t_begin, t_end, linear_depth, sun_dir, jitter);

		// out_albedo = atmosphere.rgb;
		// out_alpha = atmosphere.a;

		vec2 cloud_coverage_rotation_x = u_pc_params.cloud_coverage_rotation_x;
		mat2 cloud_coverage_rotation = mat2(cloud_coverage_rotation_x, vec2_rotate_90(cloud_coverage_rotation_x));

        CloudSettings cs;
        cs.bottom_height =            u_params.planet_radius + u_params.cloud_bottom * u_params.atmosphere_height;
        cs.top_height =               u_params.planet_radius + u_params.cloud_top * u_params.atmosphere_height;
        cs.density_scale =            u_params.cloud_density_scale;
        cs.light_density_scale =      u_params.cloud_light_density_scale;
        cs.light_reach =              u_params.cloud_light_reach;
        cs.ground_height =            u_params.planet_radius;
        cs.coverage_rotation =        cloud_coverage_rotation;
        cs.coverage_factor =          u_params.cloud_coverage_factor;
        cs.coverage_bias =            u_params.cloud_coverage_bias;
        cs.shape_factor =             u_params.cloud_shape_factor;
        cs.shape_bias =               u_params.cloud_shape_bias;
        cs.shape_scale =              u_params.cloud_shape_scale;
        cs.shape_amount =             u_params.cloud_shape_amount;
        cs.detail_factor =            u_params.cloud_detail_factor;
        cs.detail_bias =              u_params.cloud_detail_bias;
        cs.detail_scale =             u_params.cloud_detail_scale;
        cs.detail_amount =            u_params.cloud_detail_amount;
        cs.detail_falloff_distance =  u_params.cloud_detail_falloff_distance;

        cr = render_clouds(
            u_pc_params.planet_center_viewspace.xyz,
            ray_origin,
            ray_dir,
            linear_depth,
            u_cam_params.inv_view_matrix,
            u_params.world_to_model_matrix,
            sun_dir_viewspace,
            jitter,
            time,
            cs,
			vec4(
				u_params.point_light_pos_x,
				u_params.point_light_pos_y,
				u_params.point_light_pos_z,
				u_params.point_light_radius
			),
			u_params.night_light_energy
        );

	}

#ifdef FULL_RES
    vec4 color = imageLoad(u_color_image, fragcoord);
	const float exposure = 1.0;
	color.rgb = color.rgb * cr.transmittance + cr.scattering * exposure;
    imageStore(u_color_image, fragcoord, color);

#else
	imageStore(u_output_image0, fragcoord, vec4(cr.transmittance, sqrt(cr.scattering.r)));
	imageStore(u_output_image1, fragcoord, vec4(sqrt(cr.scattering.gb), 0.0, 0.0));

#endif
}
