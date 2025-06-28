@tool
class_name PlanetAtmosphereEffect
extends CompositorEffect

# Refs:
# https://www.youtube.com/watch?v=hqhWR0CxZHA

@export_group("Atmosphere")
@export var model_transform := Transform3D()
@export var planet_radius := 100.0
@export var atmosphere_height := 20.0
@export var atmosphere_density := 0.2
@export_range(0.0, 1.0, 0.01) var atmosphere_scattering_strength := 1.0

@export_group("Light sources")
@export var sun_direction := Vector3(0.0, -1.0, 0.0)
@export var night_light_energy := 0.005

@export_group("Clouds height", "clouds")
@export_range(0.0, 1.0, 0.001) var clouds_bottom := 0.3
@export_range(0.0, 1.0, 0.001) var clouds_top := 1.0
@export var clouds_density_scale := 25.0

@export_group("Clouds light", "clouds")
@export var clouds_light_density_scale := 1.0
@export_range(0.0, 2.0, 0.01) var clouds_light_reach := 1.0

@export_group("Clouds coverage", "clouds")
@export_range(0.0, 1.0) var clouds_coverage_factor := 1.0
@export var clouds_coverage_bias := -0.2
#var cloud_coverage_rotation_x := Vector2(1.0, 0.0)

@export_group("Clouds shape", "clouds")
@export var clouds_shape_enabled := true
@export_range(0.0, 1.0) var clouds_shape_factor := 0.6
@export var clouds_shape_bias := 0.0
@export var clouds_shape_scale := 0.05
@export_range(0.0, 1.0) var clouds_shape_amount := 0.5

@export_group("Clouds detail", "clouds")
@export var clouds_detail_enabled := true
@export_range(0.0, 1.0) var clouds_detail_factor := 0.6
@export var clouds_detail_bias := 0.0
@export var clouds_detail_scale := 0.05
@export_range(0.0, 1.0) var clouds_detail_amount := 0.5
@export var clouds_detail_falloff_distance := 100.0

@export_group("Debug")
@export var debug_value := 0.0;


class PointLightInfo:
	var position := Vector3()
	var radius := 10.0
	#var enabled := true


class OpticalDepth:
	const _shader_file = preload("./optical_depth.glsl")
	const _resolution = Vector2i(256, 256)

	var dirty := true
	var shader_dirty := true

	var _shader_rid := RID()
	var _pipeline_rid := RID()
	var _texture_rid := RID()


	func init(rd: RenderingDevice) -> void:
		var format0 := RDTextureFormat.new()
		format0.width = _resolution.x
		format0.height = _resolution.y
		format0.format = RenderingDevice.DATA_FORMAT_R32_SFLOAT
		format0.usage_bits = \
			RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | \
			RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
		_texture_rid = rd.texture_create(format0, RDTextureView.new(), [])
	

	func get_texture_rid() -> RID:
		return _texture_rid


	func update_shader(rd: RenderingDevice) -> bool:
		if not shader_dirty:
			return true
		shader_dirty = false

		var shader_spirv := _shader_file.get_spirv()
		if shader_spirv.compile_error_compute != "":
			return false
		_shader_rid = rd.shader_create_from_spirv(shader_spirv)
		if not _shader_rid.is_valid():
			return false
		_pipeline_rid = rd.compute_pipeline_create(_shader_rid)
		if not _pipeline_rid.is_valid():
			return false
		
		return true


	func deinit(rd: RenderingDevice) -> void:
		if _shader_rid.is_valid():
			rd.free_rid(_shader_rid)
			_shader_rid = RID()

		if _texture_rid.is_valid():
			rd.free_rid(_texture_rid)
			_texture_rid = RID()


	func render(rd: RenderingDevice, fx: PlanetAtmosphereEffect) -> void:
		print("Rendering optical depth")

		var cs_groups := Vector3i(
			PlanetAtmosphereEffect.ceildiv(_resolution.x, 8),
			PlanetAtmosphereEffect.ceildiv(_resolution.y, 8),
			1
		)

		var push_constant = PackedFloat32Array()
		push_constant.append(_resolution.x)
		push_constant.append(_resolution.y)
		push_constant.append(fx.planet_radius)
		push_constant.append(fx.atmosphere_height)
		push_constant.append(fx.atmosphere_density)
		push_constant.append(0.0) # Padding
		push_constant.append(0.0) # Padding
		push_constant.append(0.0) # Padding

		var output_image_uniform = RDUniform.new()
		output_image_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		output_image_uniform.binding = 0
		output_image_uniform.add_id(_texture_rid)

		var uniform_set_items: Array[RDUniform] = [
			output_image_uniform
		]
		
		var uniform_set := UniformSetCacheRD.get_cache(_shader_rid, 0, uniform_set_items)
		
		var compute_list := rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(compute_list, _pipeline_rid)
		rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
		var push_constant_bytes := push_constant.to_byte_array()
		rd.compute_list_set_push_constant(compute_list, push_constant_bytes, push_constant_bytes.size())
		rd.compute_list_dispatch(compute_list, cs_groups.x, cs_groups.y, cs_groups.z)
		# rd.compute_list_add_barrier(compute_list)
		rd.compute_list_end()


var _dirty: bool = true
var _mutex: Mutex

var _rd: RenderingDevice

var _shader_rid: RID
var _pipeline_rid: RID

var _post_shader_rid: RID
var _post_pipeline_rid: RID

var _linear_sampler: RID
var _linear_sampler_clamp: RID
var _nearest_sampler: RID

var _params_ubo: RID
var _cam_params_ubo: RID
var _cloud_buffer0: RID
var _cloud_buffer1: RID

var _frame_counter := 0
var _point_light := PointLightInfo.new()
var _last_screen_resolution := Vector2i()

var _optical_depth := OpticalDepth.new()

const _shader_file = preload("./effect.glsl")
const _post_shader_file = preload("./post.glsl")
const _cloud_coverage_cubemap = preload("res://tests/cloud_coverage_2.png")
const _cloud_shape_texture = preload("res://tests/swirly.tres")
const _cloud_detail_texture = preload("res://addons/zylann.atmosphere/demo/cloud_shape_texture3d.tres")
const _blue_noise_texture = preload("res://addons/zylann.atmosphere/blue_noise.png")
const _white_texture = preload("res://addons/zylann.atmosphere/white.png")

const _callback_type := EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT
const _full_res := false

func _init() -> void:
	_mutex = Mutex.new()
	
	effect_callback_type = _callback_type
	_rd = RenderingServer.get_rendering_device()
	
	RenderingServer.call_on_render_thread(_init_render)


func set_point_light(pos: Vector3, radius: float) -> void:
	_point_light.position = pos
	_point_light.radius = radius


func _init_render() -> void:
	var ss = RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	ss.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss.repeat_w = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	_nearest_sampler = _rd.sampler_create(ss)
	
	ss = RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss.repeat_w = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	_linear_sampler = _rd.sampler_create(ss)

	ss = RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER
	ss.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER
	ss.repeat_w = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER
	_linear_sampler_clamp = _rd.sampler_create(ss)

	var params_f32 := _make_params_f32()
	_params_ubo = _rd.uniform_buffer_create(params_f32.size() * 4, params_f32.to_byte_array())
	
	var cam_params_f32 := _make_camera_params_f32(null)
	_cam_params_ubo = _rd.uniform_buffer_create(
		cam_params_f32.size() * 4,
		cam_params_f32.to_byte_array()
	)

	_optical_depth.init(_rd)


func _clear_render() -> void:
	if _shader_rid.is_valid():
		_rd.free_rid(_shader_rid)
		_shader_rid = RID()

	if _post_shader_rid.is_valid():
		_rd.free_rid(_post_shader_rid)
		_shader_rid = RID()

	if _nearest_sampler.is_valid():
		_rd.free_rid(_nearest_sampler)
		_nearest_sampler = RID()
	
	if _linear_sampler.is_valid():
		_rd.free_rid(_linear_sampler)
		_linear_sampler = RID()
	
	if _params_ubo.is_valid():
		_rd.free_rid(_params_ubo)
		_params_ubo = RID()
	
	if _cam_params_ubo.is_valid():
		_rd.free_rid(_cam_params_ubo)
		_cam_params_ubo = RID()
	
	_clear_cloud_buffers()

	_optical_depth.deinit(_rd)


func _clear_cloud_buffers() -> void:
	if _cloud_buffer0.is_valid():
		_rd.free_rid(_cloud_buffer0)
		_cloud_buffer0 = RID()

	if _cloud_buffer1.is_valid():
		_rd.free_rid(_cloud_buffer1)
		_cloud_buffer1 = RID()


func _notification(what: int) -> void:
	match what:
		NOTIFICATION_PREDELETE:
			# TODO Can't call our own methods on cleanup... so we have code repetition...
			# https://github.com/godotengine/godot-proposals/issues/11774
			#_clear_render()
			_rd.free_rid(_shader_rid)
			
			if _post_shader_rid.is_valid():
				_rd.free_rid(_post_shader_rid)
			
			_rd.free_rid(_nearest_sampler)
			_rd.free_rid(_linear_sampler)
			_rd.free_rid(_params_ubo)
			_rd.free_rid(_cam_params_ubo)
			
			if _cloud_buffer0.is_valid():
				_rd.free_rid(_cloud_buffer0)

			if _cloud_buffer1.is_valid():
				_rd.free_rid(_cloud_buffer1)

			_optical_depth.deinit(_rd)


func _update_shader() -> bool:
	if not _dirty:
		return true
	_dirty = false
	
	# Does any of this actually need mutex lock?
	
	var shader_spirv := _shader_file.get_spirv()
	if shader_spirv.compile_error_compute != "":
		return false
	_shader_rid = _rd.shader_create_from_spirv(shader_spirv)
	if not _shader_rid.is_valid():
		return false
	_pipeline_rid = _rd.compute_pipeline_create(_shader_rid)
	if not _pipeline_rid.is_valid():
		return false

	var post_shader_spirv := _post_shader_file.get_spirv()
	if post_shader_spirv.compile_error_compute != "":
		return false
	_post_shader_rid = _rd.shader_create_from_spirv(post_shader_spirv)
	if not _post_shader_rid.is_valid():
		return false
	_post_pipeline_rid = _rd.compute_pipeline_create(_post_shader_rid)
	if not _post_pipeline_rid.is_valid():
		return false
	
	return true


func _render_callback(p_effect_callback_type: int, p_render_data: RenderData) -> void:
	if _rd == null:
		return
	
	# Looks redundant, but users might be able to change that in the UI
	if p_effect_callback_type != _callback_type:
		return

	if not _update_shader():
		return
	if not _optical_depth.update_shader(_rd):
		return

	_frame_counter += 1

	if _optical_depth.dirty:
		_optical_depth.dirty = false
		_optical_depth.render(_rd, self)

	# Get our render scene buffers object, this gives us access to our render buffers.
	# Note that implementation differs per renderer hence the need for the cast.
	var render_scene_buffers: RenderSceneBuffersRD = p_render_data.get_render_scene_buffers()
	if render_scene_buffers == null:
		return

	# Get our render size, this is the 3D render resolution!
	var size := render_scene_buffers.get_internal_size()
	if size.x == 0 and size.y == 0:
		return
	
	var cloud_cs_groups := Vector3i()
	
	var post_cs_groups := Vector3i(
		ceildiv(size.x, 8),
		ceildiv(size.y, 8),
		1
	)
	
	var downsample_factor := 2
	var cloud_buffer_res := size / downsample_factor
	
	if _full_res:
		cloud_cs_groups = post_cs_groups
	else:
		cloud_cs_groups.x = ceildiv(cloud_buffer_res.x, 8)
		cloud_cs_groups.y = ceildiv(cloud_buffer_res.y, 8)
		cloud_cs_groups.z = 1
		
		if size != _last_screen_resolution:
			_last_screen_resolution = size
			
			_clear_cloud_buffers()
			
			var format0 := RDTextureFormat.new()
			format0.width = cloud_buffer_res.x
			format0.height = cloud_buffer_res.y
			format0.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
			format0.usage_bits = \
				RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | \
				RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
			_cloud_buffer0 = _rd.texture_create(format0, RDTextureView.new(), [])

			var format1 := RDTextureFormat.new()
			format1.width = cloud_buffer_res.x
			format1.height = cloud_buffer_res.y
			format1.format = RenderingDevice.DATA_FORMAT_R8G8_UNORM
			format1.usage_bits = \
				RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | \
				RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
			_cloud_buffer1 = _rd.texture_create(format1, RDTextureView.new(), [])

	var time_seconds := Time.get_ticks_msec() / 1000.0

	var scene_data := p_render_data.get_render_scene_data()
	var camera_transform := scene_data.get_cam_transform()
	var world_to_view := camera_transform.inverse()
	var planet_center_viewspace := world_to_view * model_transform.origin
	var sphere_depth_factor := 0.0
	
	# TODO Sun should be a direction
	var sun_center_world := \
		model_transform.origin - sun_direction * (2.0 * (planet_radius + atmosphere_height))
	var sun_center_viewspace := world_to_view * sun_center_world
	
	# Faster than UBO but typically limited in size (128 bytes minimum).
	# Also needs to be aligned to 16 bytes
	var push_constant: PackedFloat32Array = PackedFloat32Array()
	
	if _full_res:
		push_constant.push_back(size.x)
		push_constant.push_back(size.y)
	else:
		push_constant.push_back(cloud_buffer_res.x)
		push_constant.push_back(cloud_buffer_res.y)
		
	push_constant.push_back(time_seconds)
	push_constant.push_back(_frame_counter & 255)

	push_constant.push_back(planet_center_viewspace.x)
	push_constant.push_back(planet_center_viewspace.y)
	push_constant.push_back(planet_center_viewspace.z)
	push_constant.push_back(sphere_depth_factor)

	push_constant.push_back(sun_center_viewspace.x)
	push_constant.push_back(sun_center_viewspace.y)
	push_constant.push_back(sun_center_viewspace.z)
	push_constant.push_back(0.0)
	
	var coverage_rotation_x := Vector2(1.0, 0.0)
	push_constant.push_back(coverage_rotation_x.x)
	push_constant.push_back(coverage_rotation_x.y)
	push_constant.push_back(debug_value)
	push_constant.push_back(0.0)
	
	var post_push_constant := PackedFloat32Array()
	if not _full_res:
		post_push_constant.push_back(size.x)
		post_push_constant.push_back(size.y)
		post_push_constant.push_back(cloud_buffer_res.x)
		post_push_constant.push_back(cloud_buffer_res.y)
	
	var cam_params_f32 := _make_camera_params_f32(scene_data)
	var cam_params_bytes := cam_params_f32.to_byte_array()
	_rd.buffer_update(_cam_params_ubo, 0, cam_params_bytes.size(), cam_params_bytes)
	
	# TODO Do this only if changed
	var params_f32 := _make_params_f32()
	var params_bytes := params_f32.to_byte_array()
	_rd.buffer_update(_params_ubo, 0, params_bytes.size(), params_bytes)
	
	# # Loop through views just in case we're doing stereo rendering.
	var view_count := render_scene_buffers.get_view_count()
	for view_index in view_count:
		var color_image_uniform: RDUniform
		var cloud_buffer0_uniform: RDUniform
		var cloud_buffer1_uniform: RDUniform
		
		var color_image := render_scene_buffers.get_color_layer(view_index)
		color_image_uniform = RDUniform.new()
		color_image_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		color_image_uniform.binding = 0
		color_image_uniform.add_id(color_image)
		
		if not _full_res:
			cloud_buffer0_uniform = RDUniform.new()
			cloud_buffer0_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
			cloud_buffer0_uniform.binding = 0
			cloud_buffer0_uniform.add_id(_cloud_buffer0)

			cloud_buffer1_uniform = RDUniform.new()
			cloud_buffer1_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
			cloud_buffer1_uniform.binding = 1
			cloud_buffer1_uniform.add_id(_cloud_buffer1)
		
		var depth_image := render_scene_buffers.get_depth_layer(view_index)
		var depth_texture_uniform := RDUniform.new()
		depth_texture_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		depth_texture_uniform.binding = 2
		depth_texture_uniform.add_id(_nearest_sampler)
		depth_texture_uniform.add_id(depth_image)
		
		var cloud_coverage_cubemap_rd := \
			RenderingServer.texture_get_rd_texture(_cloud_coverage_cubemap.get_rid())
		var cloud_coverage_cubemap_uniform := RDUniform.new()
		cloud_coverage_cubemap_uniform.uniform_type = \
			RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		cloud_coverage_cubemap_uniform.binding = 3
		cloud_coverage_cubemap_uniform.add_id(_linear_sampler)
		cloud_coverage_cubemap_uniform.add_id(cloud_coverage_cubemap_rd)
		
		var cloud_shape_texture_rd := \
			RenderingServer.texture_get_rd_texture(_cloud_shape_texture.get_rid())
		var cloud_shape_texture_uniform := RDUniform.new()
		cloud_shape_texture_uniform.uniform_type = \
			RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		cloud_shape_texture_uniform.binding = 4
		cloud_shape_texture_uniform.add_id(_linear_sampler)
		cloud_shape_texture_uniform.add_id(cloud_shape_texture_rd)

		var cloud_detail_texture_rd := \
			RenderingServer.texture_get_rd_texture(_cloud_detail_texture.get_rid())
		var cloud_detail_texture_uniform := RDUniform.new()
		cloud_detail_texture_uniform.uniform_type = \
			RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		cloud_detail_texture_uniform.binding = 5
		cloud_detail_texture_uniform.add_id(_linear_sampler)
		cloud_detail_texture_uniform.add_id(cloud_detail_texture_rd)
		
		var blue_noise_texture_rd := \
			RenderingServer.texture_get_rd_texture(_blue_noise_texture.get_rid())
		var blue_noise_texture_uniform := RDUniform.new()
		blue_noise_texture_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		blue_noise_texture_uniform.binding = 6
		blue_noise_texture_uniform.add_id(_linear_sampler)
		blue_noise_texture_uniform.add_id(blue_noise_texture_rd)

		var optical_depth_texture_uniform := RDUniform.new()
		optical_depth_texture_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		optical_depth_texture_uniform.binding = 7
		optical_depth_texture_uniform.add_id(_linear_sampler_clamp)
		# if _frame_counter < 10:
		# 	var ph := RenderingServer.texture_get_rd_texture(_white_texture.get_rid())
		# 	optical_depth_texture_uniform.add_id(ph)
		# else:
		optical_depth_texture_uniform.add_id(_optical_depth.get_texture_rid())

		var params_uniform := RDUniform.new()
		params_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
		params_uniform.binding = 8
		params_uniform.add_id(_params_ubo)

		var cam_params_uniform := RDUniform.new()
		cam_params_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
		cam_params_uniform.binding = 9
		cam_params_uniform.add_id(_cam_params_ubo)
		
		var uniform_set_items: Array[RDUniform]
		
		if _full_res:
			uniform_set_items = [
				color_image_uniform,
				depth_texture_uniform,
				cloud_coverage_cubemap_uniform,
				cloud_shape_texture_uniform,
				cloud_detail_texture_uniform,
				blue_noise_texture_uniform,
				optical_depth_texture_uniform,
				params_uniform,
				cam_params_uniform
			]
		else:
			uniform_set_items = [
				cloud_buffer0_uniform,
				cloud_buffer1_uniform,
				depth_texture_uniform,
				cloud_coverage_cubemap_uniform,
				cloud_shape_texture_uniform,
				cloud_detail_texture_uniform,
				blue_noise_texture_uniform,
				optical_depth_texture_uniform,
				params_uniform,
				cam_params_uniform
			]
		
		var post_uniform_set: RID
		if not _full_res:
			var depth_texture_uniform2 := RDUniform.new()
			depth_texture_uniform2.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
			depth_texture_uniform2.binding = 1
			depth_texture_uniform2.add_id(_nearest_sampler)
			depth_texture_uniform2.add_id(depth_image)
			
			var input_cloud0_uniform := RDUniform.new()
			input_cloud0_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
			input_cloud0_uniform.binding = 2
			input_cloud0_uniform.add_id(_linear_sampler)
			input_cloud0_uniform.add_id(_cloud_buffer0)

			var input_cloud1_uniform := RDUniform.new()
			input_cloud1_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
			input_cloud1_uniform.binding = 3
			input_cloud1_uniform.add_id(_linear_sampler)
			input_cloud1_uniform.add_id(_cloud_buffer1)
			
			var post_uniform_set_items = [
				color_image_uniform,
				depth_texture_uniform2,
				input_cloud0_uniform,
				input_cloud1_uniform
			]
			
			post_uniform_set = UniformSetCacheRD.get_cache(_post_shader_rid, 0, post_uniform_set_items)
		
		var uniform_set := UniformSetCacheRD.get_cache(_shader_rid, 0, uniform_set_items)
		
		var compute_list := _rd.compute_list_begin()
		_rd.compute_list_bind_compute_pipeline(compute_list, _pipeline_rid)
		_rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
		var push_constant_bytes := push_constant.to_byte_array()
		_rd.compute_list_set_push_constant(
			compute_list,
			push_constant_bytes,
			push_constant_bytes.size()
		)
		_rd.compute_list_dispatch(
			compute_list,
			cloud_cs_groups.x,
			cloud_cs_groups.y,
			cloud_cs_groups.z
		)
		_rd.compute_list_end()

		if not _full_res:
			var post_compute_list := _rd.compute_list_begin()
			_rd.compute_list_bind_compute_pipeline(post_compute_list, _post_pipeline_rid)
			_rd.compute_list_bind_uniform_set(post_compute_list, post_uniform_set, 0)
			var post_push_constant_bytes := post_push_constant.to_byte_array()
			_rd.compute_list_set_push_constant(
				post_compute_list,
				post_push_constant_bytes,
				post_push_constant_bytes.size()
			)
			_rd.compute_list_dispatch(
				post_compute_list,
				post_cs_groups.x,
				post_cs_groups.y,
				post_cs_groups.z
			)
			_rd.compute_list_end()


func _make_params_f32() -> PackedFloat32Array:
	var world_to_model := model_transform.inverse()
	
	var params_f32 := PackedFloat32Array()
	_encode_transform_to_mat4(params_f32, world_to_model)

	params_f32.append(planet_radius)
	params_f32.append(atmosphere_height)
	params_f32.append(atmosphere_density)
	params_f32.append(atmosphere_scattering_strength)
	params_f32.append(clouds_density_scale)
	params_f32.append(clouds_light_density_scale)
	params_f32.append(clouds_light_reach)
	params_f32.append(clouds_bottom)
	params_f32.append(clouds_top)
	params_f32.append(clouds_coverage_factor)
	params_f32.append(clouds_coverage_bias)
	params_f32.append(clouds_shape_factor if clouds_shape_enabled else 0.0)
	params_f32.append(clouds_shape_bias)
	params_f32.append(clouds_shape_scale)
	params_f32.append(clouds_shape_amount)
	params_f32.append(clouds_detail_factor if clouds_detail_enabled else 0.0)
	params_f32.append(clouds_detail_bias)
	params_f32.append(clouds_detail_scale)
	params_f32.append(clouds_detail_amount)
	params_f32.append(clouds_detail_falloff_distance)
	
	var pl_pos := world_to_model * _point_light.position
	#print("PL ", pl_pos, ", ", _point_light.radius)
	params_f32.append(pl_pos.x)
	params_f32.append(pl_pos.y)
	params_f32.append(pl_pos.z)
	params_f32.append(_point_light.radius)
	
	params_f32.append(night_light_energy)

	params_f32.append(0.0)
	params_f32.append(0.0)
	params_f32.append(0.0)

	#assert(params_f32.size() % 16 == 0)
	
	return params_f32


func _make_camera_params_f32(sd: RenderSceneData) -> PackedFloat32Array:
	var projection := sd.get_cam_projection() if sd != null else Projection()
	var inv_view := sd.get_cam_transform() if sd != null else Transform3D()
	var inv_projection := projection.inverse()
	
	var data_f32 := PackedFloat32Array()
	
	_encode_transform_to_mat4(data_f32, inv_view)
	_encode_projection_to_mat4(data_f32, inv_projection)
	
	return data_f32


static func _encode_transform_to_mat4(dst: PackedFloat32Array, t: Transform3D) -> void:
	var b := t.basis
	var o := t.origin

	dst.append(b.x.x)
	dst.append(b.x.y)
	dst.append(b.x.z)
	dst.append(0.0)
	
	dst.append(b.y.x)
	dst.append(b.y.y)
	dst.append(b.y.z)
	dst.append(0.0)

	dst.append(b.z.x)
	dst.append(b.z.y)
	dst.append(b.z.z)
	dst.append(0.0)

	dst.append(o.x)
	dst.append(o.y)
	dst.append(o.z)
	dst.append(1.0)


static func _encode_projection_to_mat4(dst: PackedFloat32Array, p: Projection) -> void:
	dst.append(p.x.x)
	dst.append(p.x.y)
	dst.append(p.x.z)
	dst.append(p.x.w)

	dst.append(p.y.x)
	dst.append(p.y.y)
	dst.append(p.y.z)
	dst.append(p.y.w)

	dst.append(p.z.x)
	dst.append(p.z.y)
	dst.append(p.z.z)
	dst.append(p.z.w)

	dst.append(p.w.x)
	dst.append(p.w.y)
	dst.append(p.w.z)
	dst.append(p.w.w)


static func ceildiv(x: int, d: int) -> int:
	assert(d > 0)
	if x > 0:
		return (x + d - 1) / d
	else:
		return x / d
