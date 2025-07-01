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

//#define FULL_RES
#define ENABLE_CLOUDS
#define ENABLE_ATMO

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#ifdef FULL_RES
// Rendered scene so far
layout(rgba16f, set = 0, binding = 0) uniform image2D u_input_image;
#else
layout(rgba8, set = 0, binding = 0) uniform image2D u_output_image0;
layout(rgba8, set = 0, binding = 1) uniform image2D u_output_image1;
layout(rgba8, set = 0, binding = 2) uniform image2D u_output_image2;
layout(r32f, set = 0, binding = 3) uniform image2D u_output_image3;
#endif

// Depth of the rendered scene so far
layout(binding = 4) uniform sampler2D u_depth_texture;

// Grayscale cubemap weighting overall cloud density
layout(binding = 5) uniform samplerCube u_cloud_coverage_cubemap;
// Precomputed noise used to shape the clouds, tiling seamlessly
layout(binding = 6) uniform sampler3D u_cloud_shape_texture;
layout(binding = 7) uniform sampler3D u_cloud_detail_texture;
// Blue noise used for dithering
layout(binding = 8) uniform sampler2D u_blue_noise_texture;

layout(binding = 9) uniform sampler2D u_optical_depth_texture;

// Parameters that don't change every frame
layout (binding = 10) uniform Params {
    mat4 world_to_model_matrix;
    
	float planet_radius;
    float atmosphere_height;
	float atmosphere_density;
	float atmosphere_scattering_strength;

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

	float cloud_scattering_r;
	float cloud_scattering_g;
	float cloud_scattering_b;

	float cloud_rough_steps;
	float cloud_sub_steps;
	float cloud_main_light_steps;
	float cloud_secondary_light_steps;

	float atmo_steps;
	float cloud_gamma_correction;
	float reserved1;
	float reserved2;

	float cloud_sunset_offset_r;
	float cloud_sunset_offset_g;
	float cloud_sunset_offset_b;
	float cloud_sunset_sharpness;
} u_params;

// Camera
layout (binding = 11) uniform CamParams {
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
    float debug_value; // 56..59
    float reserved2; // 60..63

	vec2 screen_size;
    float reserved3;
    float reserved4;
} u_pc_params;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const float PI = 3.14159265359;

// x = first hit, y = second hit. Equal if not hit.
// Either hits can be behind the ray's origin, as negative distance. For example if the sphere is behind, 
// or if the ray's origin is inside it.
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

float min_vec3_component(vec3 v) {
	return min(v.x, min(v.y, v.z));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Atmosphere
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct AtmosphereSettings {
	int steps;
	float planet_radius;
	float height;
	float density;
	vec3 ambient_color;
	vec3 modulate;
	vec3 scattering_wavelengths;
	float scattering_strength;
};

float get_atmosphere_density(float height, AtmosphereSettings atmo) {
	float sd = height - atmo.planet_radius;
	float h = clamp(sd / atmo.height, 0.0, 1.0);
	float y = 1.0 - h;

	float density = y * y * y * atmo.density;

	// Attenuates atmosphere in a radius around the camera
//	float distance_from_ray_origin = 0.0;
//	density *= min(1.0, (1.0 / u_attenuation_distance) * distance_from_ray_origin);

	return density;
}

float get_baked_optical_depth(
	vec3 pos,
	vec3 dir,
	vec3 planet_center,
	sampler2D optical_depth_texture,
	AtmosphereSettings atmo
) {
	float height = distance(pos, planet_center) - atmo.planet_radius;
	float height_ratio = clamp(height / atmo.height, 0.0, 1.0);
	vec3 up = normalize(pos - planet_center);
	float uvx = 0.5 + 0.5 * dot(up, dir);

	// TODO Account for samples taken below planet surface
	// they can be added as a linear equation since density is constant

	return pow2(texture(optical_depth_texture, vec2(uvx, height_ratio)).r);
}

struct AtmoResult {
	float transmittance;
	vec3 scattering;
};

AtmoResult default_atmo_result() {
	return AtmoResult(1.0, vec3(0.0));
}

AtmoResult compute_atmosphere(
	vec3 ray_origin,
	vec3 ray_dir,
	vec3 planet_center,
	float t_begin,
	float t_end,
	// float linear_depth,
	vec3 sun_dir,
	float jitter,
	AtmosphereSettings atmo
) {
	// Rocky planets don't need many steps (8 can be enough).
	// However gas giants need a lot more (64?) since rays traverse it fully.
	const int steps = atmo.steps;

	// TODO Compute this in script?
	vec3 scattering_coefficients = vec3(
		pow4(400.0 / atmo.scattering_wavelengths.x),
		pow4(400.0 / atmo.scattering_wavelengths.y),
		pow4(400.0 / atmo.scattering_wavelengths.z)
	);
	scattering_coefficients = mix(vec3(1.0), scattering_coefficients, atmo.scattering_strength);

	float step_len = (t_end - t_begin) / float(steps);

	float total_transmittance = 1.0;
	vec3 total_light = vec3(0.0);

	float total_transmittance_min = 1.0;
	vec3 total_light_min = vec3(0.0);

	float view_ray_optical_depth = 0.0;
	// float alpha = 0.0;
	vec3 pos0 = ray_origin + ray_dir * t_begin;
	vec3 pos = pos0;

	float distance_travelled = t_begin;

	for (int i = 0; i < steps; ++i) {
		float sun_ray_optical_depth = get_baked_optical_depth(
			pos, 
			sun_dir, 
			planet_center, 
			u_optical_depth_texture, 
			atmo
		);

		float height = distance(pos, planet_center);
		float local_density = get_atmosphere_density(height, atmo);
		view_ray_optical_depth += local_density * step_len;

		vec3 transmittance = exp(
			-(sun_ray_optical_depth + view_ray_optical_depth)
			* scattering_coefficients);

		total_light += local_density * step_len * transmittance * scattering_coefficients;

		float vtransmittance = exp(-local_density * step_len);
		total_transmittance *= vtransmittance;

		pos += ray_dir * step_len;
		distance_travelled += step_len;
	}

	total_light = clamp(total_light + atmo.ambient_color, vec3(0.0), vec3(1.0));
	total_light *= atmo.modulate;
	// Get rid of color banding.
	// Make sure it doesn't go out of bounds.
	// Also for some reason clamping to 1.0 caused HDR sunsets to be noisy
	total_transmittance = clamp(total_transmittance + jitter * 0.02, 0.0, 0.99);

	return AtmoResult(total_transmittance, total_light);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Clouds
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Settings carried around in cloud functions.
// We don't use uniforms directly to make the code a bit more portable.
struct CloudSettings {
	int rough_steps;
	int sub_steps;
	int main_light_steps;
	int secondary_light_steps;

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

	vec3 scattering_coefficients;

	vec3 sunset_offsets;
	float sunset_sharpness;
};

vec3 planet_shadow_curve_vec3(vec3 v) {
	return smoothstep(vec3(0.0), vec3(1.0), v);
}

// For each color component, returns an approximation of how much sun light they should get, around the terminator
vec3 get_planet_shadow(vec3 pos, vec3 sun_dir, CloudSettings settings) {
	// Offset based on height, so as we progress towards the dark side,
	// higher clouds get light for longer than lower ones
	float height_ratio = (length(pos) - settings.bottom_height) / (settings.top_height - settings.bottom_height);
	float offset_towards_dark_side = height_ratio;

	float ss = settings.sunset_sharpness;
	float dp = dot(normalize(pos), sun_dir);
	vec3 sv = vec3(dp * ss) + settings.sunset_offsets + vec3(offset_towards_dark_side);
	return planet_shadow_curve_vec3(sv);
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

// Adapted from: https://twitter.com/FewesW/status/1364629939568451587/photo/1
vec3 multi_octave_scatter(float density, float mu, vec3 scattering_coefficients) {
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
		vec3 beers = exp(-density * scattering_coefficients * a);

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

	vec3 beers_law = multi_octave_scatter(total_density, mu, settings.scattering_coefficients);
	vec3 powder = 1.0 - exp(-total_density * 2.0 * settings.scattering_coefficients);
	
	return beers_law * mix(2.0 * powder, vec3(1.0), mu * 0.5 + 0.5);
}

float calculate_point_light_energy_factor(vec3 pos, vec4 point_light) {
	return 1.1 * pow4(max(1.0 - distance(point_light.xyz, pos) / point_light.w, 0.0));
}

struct CloudResult {
	vec3 scattering;
	vec3 transmittance;
	float depth;
};

const float CloudResult_MAX_DEPTH = 999999.0;

CloudResult default_cloud_result() {
	return CloudResult(vec3(0.0), vec3(1.0), CloudResult_MAX_DEPTH);
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

	// {
	// 	// This is a hack limiting marching distance to increase horizon quality at certain heights.
	// 	// Without it, horizon peers too much through the cloud layer when seen from space.
	// 	// So we cut off how far we march, and gradually increase it as we descend through the clouds.
	// 	// So the worst case scenario is now while being inside the clouds, which is better than having
	// 	// that discrepancy all the time
	// 	float march_distance_space =
	// 		0.5 * sqrt(
	// 			1.0 - pow2(settings.ground_height / settings.top_height)
	// 		) * settings.bottom_height;
	// 	float march_distance_ground = 3.0 * march_distance_space;
	// 	float march_distance_transition_height_min = settings.bottom_height;
	// 	float march_distance_transition_height_max = settings.top_height * 1.05;

	// 	float max_d = mix(
	// 		march_distance_ground,
	// 		march_distance_space,
	// 		smoothstep(
	// 			march_distance_transition_height_min,
	// 			march_distance_transition_height_max,
	// 			length(ray_origin)
	// 		)
	// 	);

	// 	t_end = t_begin + min(t_end - t_begin, max_d);
	// }

	{
		float max_d = 25.0;
		t_end = t_begin + min(t_end - t_begin, max_d);
	}

	vec3 sun_color = vec3(1.0);
	const float cloud_light_multiplier = 50.0;
	vec3 sun_light = sun_color * cloud_light_multiplier;
	const float ambient_strength = 0.0;
	vec3 ambient = vec3(ambient_strength * sun_color);

	const float ray_dist = t_end - t_begin;
	// const float step_dropoff = linearstep(1.0, 0.0, pow4(dot(vec3(0.0, 1.0, 0.0), ray_dir)));

	const int rough_steps = settings.rough_steps;
	const int sub_steps = settings.sub_steps;

	float lq_step_len = ray_dist / float(rough_steps);
	float hq_step_len = lq_step_len / float(sub_steps);
	const float max_steps = rough_steps * sub_steps;
	float step_len = hq_step_len;

	// const float offset = lq_step_len * jitter;
	float dist_travelled = t_begin;
	// dist_travelled += lq_step_len * jitter;
	dist_travelled += hq_step_len * jitter;

	// int hq_marcher_countdown = 0;

	// TODO Not used?
	// float previous_step_len = 0.0;

	CloudResult result = default_cloud_result();
	const float break_transmittance = 0.01;

	// float current_step_len = lq_step_len;

	for (float i = 0.0; i < max_steps; ++i) {
		// TODO Is this really needed? We already do max steps which is the sum of all potential hq steps
		if (dist_travelled > t_end) {
			break;
		}

		vec3 pos = ray_origin + dist_travelled * ray_dir;
		Coverage sd1 = sample_sdf_low(pos, settings);

		if (sd1.combined < 0.0) {
			float extinction = sample_density(sd1, pos, cam_pos, time, settings);

			if (extinction > 0.01) {
				vec3 light_energy = vec3(0.0);

				vec3 planet_shadow = get_planet_shadow(pos, sun_dir, settings);
				float planet_shadow_max = max_vec3_component(planet_shadow);
				// float planet_shadow_min = min_vec3_component(planet_shadow);

				// Sun light
				if (planet_shadow_max > 0.0) {
					light_energy += planet_shadow * raymarch_light_energy(
						pos, 
						sun_dir, 
						cam_pos, 
						ray_dir, 
						time, 
						jitter, 
						settings, 
						settings.main_light_steps
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
						settings.secondary_light_steps
					);
				}

				// Night light
				// TODO Option to use cheap height light?
				if (planet_shadow_max < 1.0) {
					light_energy += (vec3(1.0) - planet_shadow) * night_light_energy * raymarch_light_energy(
						pos, 
						normalize(pos), 
						cam_pos, 
						ray_dir, 
						time, 
						jitter, 
						settings,
						settings.secondary_light_steps
					);
				}

				vec3 luminance = ambient + sun_light * light_energy;
				vec3 transmittance = exp(-extinction * step_len * settings.scattering_coefficients);
				vec3 integ_scatt = luminance * (1.0 - transmittance);

				result.scattering += result.transmittance * integ_scatt;
				result.transmittance *= transmittance;

				if (max_vec3_component(result.transmittance) <= break_transmittance) {
					result.transmittance = vec3(0.0);
					break;
				}

				if (extinction > 0.1) {
					// TODO Use dist_travelled?
					result.depth = min(result.depth, distance(ray_origin, pos));
				}

			} else {
				step_len = hq_step_len;
			}
		}

		dist_travelled += step_len;
	}

	const vec3 cloud_color = vec3(1.0);

	result.scattering = result.scattering * cloud_color;
	result.transmittance = clamp(result.transmittance, vec3(0.0), vec3(1.0));

	return result;
}

vec3 color_curve(vec3 v) {
	v = clamp(v, vec3(0.0), vec3(2.0));
	vec3 a = 0.5 * v - vec3(1.0);
	vec3 r = vec3(1.0) - a * a;
	return clamp(r, vec3(0.0), vec3(1.0));
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

	CloudResult result = default_cloud_result();

	if (rs_clouds_top.x != rs_clouds_top.y) {
		vec2 rs_clouds_bottom = ray_sphere(planet_center_viewspace, cloud_settings.bottom_height, ray_origin, ray_dir);

		vec2 cloud_rs = rs_clouds_top;
		cloud_rs.x = max(cloud_rs.x, 0.0);
		cloud_rs.y = min(cloud_rs.y, linear_depth);

		if (cloud_rs.x < linear_depth
			// Don't compute clouds when opaque stuff occludes them,
			// when under the clouds layer.
			// This saves 0.5ms in ground views on a 1060
			// && (linear_depth > rs_clouds_bottom.y || rs_clouds_bottom.x > 0.0)
		) {
			mat4 view_to_model_matrix = world_to_model_matrix * inv_view_matrix;
			vec3 cam_pos_model = (view_to_model_matrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
			vec3 sun_dir_model = (view_to_model_matrix * vec4(sun_dir_viewspace, 0.0)).xyz;
			vec3 ray_origin_model = (view_to_model_matrix * vec4(ray_origin, 1.0)).xyz;
			vec3 ray_dir_model = (view_to_model_matrix * vec4(ray_dir, 0.0)).xyz;

			// When under the cloud layer, this improves quality significantly,
			// unfortunately entering the cloud layer causes a jarring transition
			if (length_sq_vec3(cam_pos_model) < pow2(cloud_settings.bottom_height)) {
				cloud_rs.x = rs_clouds_bottom.y;
			}

			result = raymarch_cloud(
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
		}
	}

	return result;
}

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

float compute_linear_depth_2x2_min_max(
		ivec2 fragcoord,
		vec2 screen_res,
		vec2 low_res,
		out vec4 view_coords,
		out float result_nonlinear_depth
) {
	vec2 screen_pixel = vec2(fragcoord) * 2.0 + vec2(0.5);
	vec2 screen_ps = vec2(1.0) / screen_res;
	
	vec2 screen_uv00 = (screen_pixel + vec2(0.0, 0.0)) * screen_ps;
	vec2 screen_uv10 = (screen_pixel + vec2(1.0, 0.0)) * screen_ps;
	vec2 screen_uv01 = (screen_pixel + vec2(0.0, 1.0)) * screen_ps;
	vec2 screen_uv11 = (screen_pixel + vec2(1.0, 1.0)) * screen_ps;

	float nonlinear_depth00 = texture(u_depth_texture, screen_uv00).x;
	float nonlinear_depth10 = texture(u_depth_texture, screen_uv10).x;
	float nonlinear_depth01 = texture(u_depth_texture, screen_uv01).x;
	float nonlinear_depth11 = texture(u_depth_texture, screen_uv11).x;

	float nonlinear_depth = (((fragcoord.x + fragcoord.y) & 1) == 0) ?
		min(min(nonlinear_depth00, nonlinear_depth10), min(nonlinear_depth01, nonlinear_depth11)) :
		max(max(nonlinear_depth00, nonlinear_depth10), max(nonlinear_depth01, nonlinear_depth11));
	
	result_nonlinear_depth = nonlinear_depth;

	vec2 low_uv = (vec2(fragcoord) + vec2(0.5)) / low_res;
	return compute_linear_depth_from_nonlinear(low_uv, nonlinear_depth, view_coords);
}

void main() {
    ivec2 fragcoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = ivec2(u_pc_params.raster_size);

    if (fragcoord.x >= image_size.x || fragcoord.y >= image_size.y) {
        return;
    }

    vec4 view_coords;
	float nonlinear_depth;
    float linear_depth = compute_linear_depth_2x2_min_max(
		fragcoord, 
		u_pc_params.screen_size, 
		vec2(image_size), 
		view_coords, 
		nonlinear_depth
	);

	// We'll evaluate the atmosphere in view space
	vec3 ray_origin = vec3(0.0, 0.0, 0.0);
	vec3 ray_dir = normalize(view_coords.xyz - ray_origin);

	float atmosphere_radius = u_params.planet_radius + u_params.atmosphere_height;
	vec3 planet_center_viewspace = u_pc_params.planet_center_viewspace.xyz;
	vec2 rs_atmo = ray_sphere(planet_center_viewspace, atmosphere_radius, ray_origin, ray_dir);

	// TODO if we run this shader in a double-clip scenario,
	// we have to account for the near and far clips properly, so they can be composed seamlessly

	CloudResult cr = default_cloud_result();

	if (rs_atmo.x != rs_atmo.y) {
		float t_begin = max(rs_atmo.x, 0.0);
		float t_end = max(rs_atmo.y, 0.0);

		vec2 rs_ground = ray_sphere(
            planet_center_viewspace, 
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

		vec3 sun_dir_viewspace = normalize(u_pc_params.sun_center_viewspace.xyz - planet_center_viewspace);

		float time = u_pc_params.time;
		float frame = u_pc_params.frame;

		// Blue noise doesn't have low-frequency patterns, it looks less "noisy"
		// http://momentsingraphics.de/BlueNoise.html
		// ivec2 blue_noise_uv = (fragcoord + ivec2(time * 100.0)) & ivec2(0xff);
		ivec2 blue_noise_uv = fragcoord & ivec2(0xff);
		float jitter = texelFetch(u_blue_noise_texture, blue_noise_uv, 0).r;

		// Animating Noise For Integration Over Time
		// https://blog.demofox.org/2017/10/31/animating-noise-for-integration-over-time/		
		// const float golden_ratio = 1.61803398875;
		// jitter = fract(jitter + float(int(frame) % 32) * golden_ratio);

		AtmosphereSettings atmo;
		atmo.steps =                   int(u_params.atmo_steps);
		atmo.planet_radius =           u_params.planet_radius;
		atmo.height =                  u_params.atmosphere_height;
		atmo.density =                 u_params.atmosphere_density;
		atmo.ambient_color =           vec3(0.01);
		atmo.modulate =                vec3(1.0);
		atmo.scattering_wavelengths =  vec3(700.0, 530.0, 440.0);
		atmo.scattering_strength =     u_params.atmosphere_scattering_strength;
	
		vec2 cloud_coverage_rotation_x = u_pc_params.cloud_coverage_rotation_x;
		mat2 cloud_coverage_rotation = mat2(cloud_coverage_rotation_x, vec2_rotate_90(cloud_coverage_rotation_x));

        CloudSettings cs;
		cs.rough_steps =              int(u_params.cloud_rough_steps);
		cs.sub_steps =                int(u_params.cloud_sub_steps);
		cs.main_light_steps =         int(u_params.cloud_main_light_steps);
		cs.secondary_light_steps =    int(u_params.cloud_secondary_light_steps);
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

		cs.scattering_coefficients =  vec3(
			u_params.cloud_scattering_r,
			u_params.cloud_scattering_g,
			u_params.cloud_scattering_b
		);
		cs.sunset_offsets = vec3(
			u_params.cloud_sunset_offset_r,
			u_params.cloud_sunset_offset_g,
			u_params.cloud_sunset_offset_b
		);
		cs.sunset_sharpness =         u_params.cloud_sunset_sharpness;

		vec4 point_light = vec4(
			u_params.point_light_pos_x,
			u_params.point_light_pos_y,
			u_params.point_light_pos_z,
			u_params.point_light_radius
		);

#ifdef ENABLE_CLOUDS
        cr = render_clouds(
            planet_center_viewspace,
            ray_origin,
            ray_dir,
            linear_depth,
            u_cam_params.inv_view_matrix,
            u_params.world_to_model_matrix,
            sun_dir_viewspace,
            jitter,
            time,
            cs,
			point_light,
			u_params.night_light_energy
        );
#endif

#ifdef ENABLE_ATMO
		// AtmoIntegration atmo_integ = AtmoIntegration_init();
		AtmoResult atmo_result2 = default_atmo_result();
		AtmoResult atmo_result = compute_atmosphere(
			ray_origin, 
			ray_dir, 
			planet_center_viewspace,
			t_begin, 
			t_end, 
			sun_dir_viewspace, 
			jitter,
			atmo
		);

		cr.scattering = mix(cr.scattering, color_curve(cr.scattering), u_params.cloud_gamma_correction);

		vec3 c_transmittance = cr.transmittance;
		cr.scattering += cr.transmittance * atmo_result.scattering;
		cr.transmittance *= atmo_result.transmittance;

		if (cr.depth < CloudResult_MAX_DEPTH) {
			// On top of clouds

			AtmoResult atmo_result_front = compute_atmosphere(
				ray_origin, 
				ray_dir, 
				planet_center_viewspace,
				t_begin, 
				cr.depth, 
				sun_dir_viewspace, 
				jitter,
				atmo
			);

			cr.scattering = mix(atmo_result_front.scattering, cr.scattering, 
				mix(atmo_result_front.transmittance, 1.0, c_transmittance.r));

			// float fmin = 10.0;
			// float fmax = 40.0;
			// float f = clamp((cr.depth - t_begin - fmin) / (fmax - fmin), 0.0, 1.0);
			// cr.transmittance = mix(cr.transmittance, vec3(atmo_result.transmittance), f);
			// cr.scattering = mix(cr.scattering, atmo_result.scattering, f);
		}
#endif
	}

#ifdef FULL_RES
    vec4 color = imageLoad(u_color_image, fragcoord);
	const float exposure = 1.0;
	color.rgb = color.rgb * cr.transmittance + cr.scattering * exposure;
    imageStore(u_color_image, fragcoord, color);

#else
	// imageStore(u_output_image0, fragcoord, vec4(cr.transmittance, sqrt(cr.scattering.r)));
	// imageStore(u_output_image1, fragcoord, vec4(sqrt(cr.scattering.gb), 0.0, 0.0));

	// if (fract(u_pc_params.time) < 0.5) {
	// 	cr.scattering = cr2.scattering;
	// 	cr.transmittance = cr2.transmittance;
	// }
	// cr.depth = 0.0;
	// cr2 = cr;

	imageStore(u_output_image0, fragcoord, vec4(
		cr.transmittance.r,
		cr.transmittance.g,
		cr.transmittance.b,
		cr.scattering.r
	));
	imageStore(u_output_image1, fragcoord, vec4(
		cr.scattering.g,
		cr.scattering.b,
		0.0,
		0.0
	));
	// imageStore(u_output_image2, fragcoord, vec4(
	// 	cr2.transmittance.b,
	// 	cr2.scattering.r,
	// 	cr2.scattering.g,
	// 	cr2.scattering.b
	// ));
	imageStore(u_output_image3, fragcoord, vec4(nonlinear_depth));

	// float dep = fract(0.1*mix(linear_depth_min, linear_depth_max, fract(u_pc_params.time)));
	// imageStore(u_output_image1, fragcoord, vec4(dep, dep, 0.0, 0.0));

#endif

	// DEBUG
	// float od = texture(u_optical_depth_texture, screen_uv).r;
	// imageStore(u_output_image0, fragcoord, vec4(vec3(0.1), od)); 
	// imageStore(u_output_image1, fragcoord, vec4(0.0, 0.0, 0.0, 0.0)); 
}
