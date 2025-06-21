#[compute]
#version 450

// #define CLOUDS_RAYMARCHED_LIGHTING
// #define CLOUDS_ALWAYS_LOW_QUALITY
#define CLOUDS_MAX_RAYMARCH_STEPS 32
#define CLOUDS_MAX_RAYMARCH_FSTEPS 4
#define CLOUDS_LIGHT_RAYMARCH_STEPS 6

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Rendered scene so far
layout(rgba16f, set = 0, binding = 0) uniform image2D u_color_image;
// Depth of the rendered scene so far
layout(binding = 1) uniform sampler2D u_depth_texture;

// Grayscale cubemap weighting overall cloud density
layout(binding = 2) uniform samplerCube u_cloud_coverage_cubemap;
// Precomputed noise used to shape the clouds, tiling seamlessly
layout(binding = 3) uniform sampler3D u_cloud_shape_texture;
// Blue noise used for dithering
layout(binding = 4) uniform sampler2D u_blue_noise_texture;

// Parameters that don't change every frame
layout (binding = 5) uniform Params {
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

	float reserved0;
	float reserved1;
	float reserved2;
} u_params;

// Camera
layout (binding = 6) uniform CamParams {
    mat4 inv_view_matrix;
    mat4 inv_projection_matrix;
} u_cam_params;

// Parameters that may change every frame
layout(push_constant, std430) uniform PcParams {
    vec2 raster_size; // 0..7
    float time; // 8..11
    float reserved0; // 12..15

    vec4 planet_center_viewspace; // 16..31 // w contains sphere_depth_factor

    vec4 sun_center_viewspace; // 32..47 // w is not used

    vec2 cloud_coverage_rotation_x; // 48..55
    float reserved1; // 56..59
    float reserved2; // 60..63
} u_pc_params;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// x = first hit, y = second hit. Equal if not hit.
vec2 ray_sphere(vec3 center, float radius, vec3 ray_origin, vec3 ray_dir) {
	// Works when outside the sphere but breaks when inside at certain positions
	/*
	float t = max(dot(center - ray_origin, ray_dir), 0.0);
	float y = length(center - (ray_origin + ray_dir * t));
	// TODO y * y means we can use a squared length
	float x = sqrt(max(radius * radius - y * y, 0.0));
	return vec2(t - x, t + x);
	*/
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
};

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

float get_density(vec3 pos_world, float time, CloudSettings settings, int quality) {
	float height = length(pos_world) - settings.bottom_height;
	float height_ratio = height / (settings.top_height - settings.bottom_height);

	if (height_ratio < 0.0 || height_ratio > 1.0) {
		return 0.0;
	}

	vec2 coverage_pos_2d = settings.coverage_rotation * pos_world.xz;
	vec3 coverage_pos = vec3(coverage_pos_2d.x, pos_world.y, coverage_pos_2d.y);
	float coverage = texture(u_cloud_coverage_cubemap, coverage_pos).r;
	coverage = clamp(coverage * settings.coverage_factor + settings.coverage_bias, 0.0, 1.0);
	coverage = band_p2s_unit(height_ratio * 2.0 - 1.0, 0.5) * coverage;
	
	if (coverage <= 0.0) {
		return 0.0;
	}

	if (quality == 0) {
		return coverage;
	}

	vec3 shape_uv = pos_world * settings.shape_scale + vec3(time*0.01, 0.0, 0.0);
	float shape = texture(u_cloud_shape_texture, shape_uv).r;
	shape = shape * settings.shape_factor + settings.shape_bias;

	float density = max(coverage - shape * settings.shape_amount, 0.0);
	if (density <= 0.0) {
		return 0.0;
	}

	if (quality == 1) {
		return density;
	}

	vec3 detail_uv = pos_world * settings.detail_scale + vec3(time*0.005, 0.0, 0.0);
	float detail = texture(u_cloud_shape_texture, detail_uv).r;
	detail = detail * settings.detail_factor + settings.detail_bias;

	density = max(density - detail * settings.detail_amount, 0.0);

	return density;
}

float get_planet_shadow(vec3 pos, float planet_radius, vec3 sun_dir) {
	float dp = clamp(dot(normalize(pos), sun_dir) * 4.0 + 0.5, 0.0, 1.0);
	return dp * dp;
	// return 1.0;
}

float get_light_raymarched(
		vec3 pos0, 
		vec3 sun_dir, 
		float jitter, // TODO Use jitter?
		float time, 
		CloudSettings settings
) {
	const int steps = CLOUDS_LIGHT_RAYMARCH_STEPS;
	float reach = (settings.top_height - settings.bottom_height) * settings.light_reach;

	float inv_steps = 1.0 / float(steps);
	float step_len = reach * inv_steps;

	pos0 += sun_dir * (step_len * jitter);

	float total_density = 0.0;
	
	for (int i = 0; i < steps; ++i) {
		vec3 pos = pos0 + float(i + 1) * step_len * sun_dir;

		float density = get_density(pos, time, settings, 1);
		density *= settings.density_scale;
		total_density += density * step_len;
	}

	total_density *= settings.density_scale * settings.light_density_scale;

	float transmittance = exp(-total_density);

	transmittance *= get_planet_shadow(pos0, settings.ground_height, sun_dir);

	return transmittance;
}

#ifndef CLOUDS_MAX_RAYMARCH_STEPS
// Need to define it otherwise the Godot shader editor has errors. Using a small value to make it
// stand out. Normally the shader using this file should define it.
#define CLOUDS_MAX_RAYMARCH_STEPS 8
#endif

#ifndef CLOUDS_MAX_RAYMARCH_FSTEPS
#define CLOUDS_MAX_RAYMARCH_FSTEPS 1
#endif

struct CloudResult {
	float transmittance;
	float light;
};

CloudResult raymarch_cloud(
	vec3 ray_origin, // in planet space
	vec3 ray_dir, 
	float t_begin, 
	float t_end, 
	float jitter,
	vec3 sun_dir, 
	float time, 
	CloudSettings settings
) {
	const int steps = CLOUDS_MAX_RAYMARCH_STEPS;

	float inv_steps = 1.0 / float(steps);
	float step_len = (t_end - t_begin) * inv_steps;

	float total_transmittance = 1.0;
	float total_light = 0.0;

	vec3 pos = ray_origin + jitter * step_len * ray_dir + ray_dir * t_begin;
	float phase = 1.0; // ?

	const int fsteps = CLOUDS_MAX_RAYMARCH_FSTEPS;
	float fstep_len = step_len / float(fsteps);

	const float opaque_transmittance = 0.01;

	for (int i = 0; i < steps; ++i) {
		float density = get_density(pos, time, settings, 0);
		
		if (density > 0.0) {
			pos -= ray_dir * step_len;

			for (int j = 0; j < fsteps; ++j) {
				float density2 = get_density(pos, time, settings, 2);

				if (density2 > 0.0) {
					density2 *= settings.density_scale;
					float light_transmittance = get_light_raymarched(pos, sun_dir, jitter, time, settings);
					total_light += density2 * fstep_len * total_transmittance * light_transmittance * phase;

					float step_transmittance = exp(-density2 * fstep_len);
					total_transmittance *= step_transmittance;

					if (total_transmittance < opaque_transmittance) {
						break;
					}
				}

				pos += ray_dir * fstep_len;
			}

			if (total_transmittance < opaque_transmittance) {
				break;
			}
		}

		pos += ray_dir * step_len;
	}

	return CloudResult(total_transmittance, total_light);
}

void render_clouds(
	inout vec4 inout_color,
	vec3 planet_center_viewspace,
	vec3 ray_origin,
	vec3 ray_dir,
	float linear_depth,
	mat4 inv_view_matrix,
    mat4 world_to_model_matrix,
	vec3 sun_dir_viewspace,
	float jitter,
    float time,
    CloudSettings cloud_settings
) {
	vec2 rs_clouds_top = ray_sphere(planet_center_viewspace, cloud_settings.top_height, ray_origin, ray_dir);

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
			// When under the cloud layer, this improves quality significantly,
			// unfortunately entering the cloud layer causes a jarring transition
			// if (rs_clouds_bottom.x < 0.0) {
			// 	cloud_rs.x = rs_clouds_bottom.y;
			// }

			mat4 view_to_model_matrix = world_to_model_matrix * inv_view_matrix;
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
                cloud_settings
            );

			vec3 light_color = vec3(1.0, 1.0, 1.0);
			vec3 cloud_color = rr.light * light_color;
			inout_color.rgb = inout_color.rgb * rr.transmittance + cloud_color;
		}
	}
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

    vec4 color = imageLoad(u_color_image, fragcoord);

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

		// Blue noise doesn't have low-frequency patterns, it looks less "noisy"
		// http://momentsingraphics.de/BlueNoise.html
		float jitter = texelFetch(u_blue_noise_texture, (fragcoord + ivec2(time * 100.0)) & ivec2(0xff), 0).r;
//		jitter = 0.0;
		// jitter = fract(time);

		// vec4 atmosphere = compute_atmosphere_v2(ray_origin, ray_dir, in_v_planet_center_viewspace,
		// 	t_begin, t_end, linear_depth, sun_dir, jitter);

		// out_albedo = atmosphere.rgb;
		// out_alpha = atmosphere.a;

		vec2 cloud_coverage_rotation_x = u_pc_params.cloud_coverage_rotation_x;
		mat2 cloud_coverage_rotation = mat2(cloud_coverage_rotation_x, vec2_rotate_90(cloud_coverage_rotation_x));

        CloudSettings cloud_settings;
        cloud_settings.bottom_height =        u_params.planet_radius + u_params.cloud_bottom * u_params.atmosphere_height;
        cloud_settings.top_height =           u_params.planet_radius + u_params.cloud_top * u_params.atmosphere_height;
        cloud_settings.density_scale =        u_params.cloud_density_scale;
        cloud_settings.light_density_scale =  u_params.cloud_light_density_scale;
        cloud_settings.light_reach =          u_params.cloud_light_reach;
        cloud_settings.ground_height =        u_params.planet_radius;
        cloud_settings.coverage_rotation =    cloud_coverage_rotation;
        cloud_settings.coverage_factor =      u_params.cloud_coverage_factor;
        cloud_settings.coverage_bias =        u_params.cloud_coverage_bias;
        cloud_settings.shape_factor =         u_params.cloud_shape_factor;
        cloud_settings.shape_bias =           u_params.cloud_shape_bias;
        cloud_settings.shape_scale =          u_params.cloud_shape_scale;
        cloud_settings.shape_amount =         u_params.cloud_shape_amount;
        cloud_settings.detail_factor =        u_params.cloud_detail_factor;
        cloud_settings.detail_bias =          u_params.cloud_detail_bias;
        cloud_settings.detail_scale =         u_params.cloud_detail_scale;
        cloud_settings.detail_amount =        u_params.cloud_detail_amount;

        render_clouds(
			color,
            u_pc_params.planet_center_viewspace.xyz,
            ray_origin,
            ray_dir,
            linear_depth,
            u_cam_params.inv_view_matrix,
            u_params.world_to_model_matrix,
            sun_dir_viewspace,
            jitter,
            time,
            cloud_settings
        );
	}
	// float nonlinear_depth = texture(u_depth_texture, screen_uv).x;

    imageStore(u_color_image, fragcoord, color);
}
