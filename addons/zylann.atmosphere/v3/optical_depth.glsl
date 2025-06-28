#[compute]
#version 450

// This shader renders optical depth in a square texture that can be used as precalculated lookup,
// so that we don't need to raymarch through a sphere to get the total travelled density.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(r32f, set = 0, binding = 0) uniform image2D u_output_image;

layout(push_constant, std430) uniform PcParams {
    vec2 raster_size;
    float planet_radius;
    float atmosphere_height;
    float atmosphere_density;

    float reserved0;
    float reserved1;
    float reserved2;
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Atmosphere funcs
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct AtmosphereSettings {
    float planet_radius;
    float height;
    float density;
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Bakes a texture where:
// X = 0.5 + 0.5 * ray_dir.y
// Y = (distance(ray_origin, planet_center) - planet_radius) / atmosphere_height
// Assuming atmosphere density only changes with height.

float get_optical_depth(vec2 ray_origin, vec2 ray_dir, float ray_len, AtmosphereSettings atmo) {
	const int steps = 64;

	float step_len = ray_len / float(steps);
	float optical_depth = 0.0;

	for (int i = 0; i < steps; ++i) {
		vec2 pos = ray_origin + ray_dir * step_len * float(i);
		float d = length(pos);
		float density = get_atmosphere_density(d, atmo);
		optical_depth += density * step_len * atmo.density;
	}

	return optical_depth;
}

void main() {
    ivec2 fragcoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(u_pc_params.raster_size);

    if (fragcoord.x >= size.x || fragcoord.y >= size.y) {
        return;
    }

    AtmosphereSettings atmo;
    atmo.planet_radius = u_pc_params.planet_radius;
    atmo.height = u_pc_params.atmosphere_height;
    atmo.density = u_pc_params.atmosphere_density;

	vec2 uv = vec2(fragcoord) / vec2(size);

	vec2 ray_dir;
	ray_dir.y = 2.0 * uv.x - 1.0;
//	ray_dir.x = sin(acos(ray_dir.y));
	ray_dir.x = sqrt(1.0 - ray_dir.y * ray_dir.y);

	float height_ratio = uv.y;

	vec2 pos = vec2(0.0, atmo.planet_radius + atmo.height * height_ratio);
	vec2 rs = ray_sphere(
		vec3(0.0),
		atmo.planet_radius + atmo.height,
		vec3(pos, 0.0),
		vec3(ray_dir, 0.0)
	);

	float distance_through_atmosphere = rs.y - max(rs.x, 0.0);

	float od = get_optical_depth(pos, ray_dir, distance_through_atmosphere, atmo);

    imageStore(u_output_image, fragcoord, vec4(od));
}
