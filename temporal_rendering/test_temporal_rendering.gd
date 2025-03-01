extends Node

const TemporalBufferShader = preload("./temporal_buffer.gdshader")
const TemporalRenderShader = preload("./temporal_render.gdshader")

@onready var _camera : Camera3D = $Avatar/Camera
@onready var _texture_rect : TextureRect = $TextureRect

var _prev_view_proj_matrix := Projection()
var _prev_camera_transform := Transform3D()

class Buffer:
	var viewport : SubViewport
	var shader_material : ShaderMaterial

class Render:
	var viewport : SubViewport
	var shader_material : ShaderMaterial

var _buffer0 := Buffer.new()
var _buffer1 := Buffer.new()
var _render := Render.new()
var _quad_mesh : QuadMesh


func _init():
	_quad_mesh = QuadMesh.new()
	_quad_mesh.size = Vector2(2, 2)
	_quad_mesh.flip_faces = true


func _ready():
	_setup_render(_render)
	_setup_buffer(_buffer0)
	_setup_buffer(_buffer1)


func _setup_buffer(b: Buffer):
	b.viewport = SubViewport.new()
	b.viewport.size = get_viewport().size
	b.viewport.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
	b.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	b.viewport.transparent_bg = false
	b.viewport.disable_3d = true
	add_child(b.viewport)
	
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.0, 0.0, 0.0)
	
	var world_env := WorldEnvironment.new()
	world_env.environment = env
	b.viewport.add_child(world_env)
	
	b.shader_material = ShaderMaterial.new()
	b.shader_material.set_shader_parameter(&"u_render_texture", _render.viewport.get_texture())
	b.shader_material.shader = TemporalBufferShader
	
	var ci := ColorRect.new()
	ci.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ci.material = b.shader_material
	
	b.viewport.add_child(ci)
	
#	var quad_mi := MeshInstance3D.new()
#	quad_mi.mesh = _quad_mesh
#	quad_mi.material_override = b.shader_material
#	quad_mi.extra_cull_margin = 100.0
#	b.viewport.add_child(quad_mi)
#
#	b.camera = Camera3D.new()
#	b.viewport.add_child(b.camera)


func _setup_render(r: Render):
	r.viewport = SubViewport.new()
	r.viewport.size = get_viewport().size / 4
	r.viewport.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
	r.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	r.viewport.transparent_bg = false
	r.viewport.disable_3d = true
	add_child(r.viewport)
	
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.0, 0.0, 0.0)
	
	var world_env := WorldEnvironment.new()
	world_env.environment = env
	r.viewport.add_child(world_env)
	
	r.shader_material = ShaderMaterial.new()
	r.shader_material.shader = TemporalRenderShader
	
	var cm := NoiseCubemap.new()
	r.shader_material.set_shader_parameter(&"u_cubemap", cm)

	var ci := ColorRect.new()
	ci.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ci.material = r.shader_material
	
	r.viewport.add_child(ci)
	
#	var quad_mi := MeshInstance3D.new()
#	quad_mi.mesh = _quad_mesh
#	quad_mi.material_override = r.shader_material
#	quad_mi.extra_cull_margin = 100.0
#	r.viewport.add_child(quad_mi)
#
#	r.camera = Camera3D.new()
#	r.viewport.add_child(r.camera)


func _process(delta):
	if Input.is_key_pressed(KEY_KP_0):
		_camera._yaw += 0.1
		_camera.update_rotations()
	
	var view_matrix := _camera.global_transform.inverse()
	
	var viewport : Viewport = get_viewport()
	# Need to use buffer aspect because it must be multiple of downscale size
	var aspect := Vector2(_buffer0.viewport.size).aspect()
	# TODO Use `Camera3D.get_projection_matrix()` in 4.2
	var proj_matrix := Projection.create_perspective(_camera.fov, aspect, _camera.near, _camera.far)
	var view_proj_matrix := proj_matrix * Projection(view_matrix)
	
	var proj_matrix_inverse := proj_matrix.inverse()
	var view_matrix_inverse := view_matrix.inverse()
	
	var frame_index := Engine.get_frames_drawn() / 1
	
#	_buffer1.camera.global_transform = _camera.global_transform
	_buffer1.shader_material.set_shader_parameter(
		&"u_prev_frame_view_projection_matrix", _prev_view_proj_matrix)
	_buffer1.shader_material.set_shader_parameter(&"u_frame_index", frame_index)
	# WHY IS SCREEN TEXTURE NEVER WORKING THE WAY I WANT
	_buffer1.shader_material.set_shader_parameter(
		&"u_prev_screen_texture", _buffer0.viewport.get_texture())
	_buffer1.shader_material.set_shader_parameter(&"u_inv_projection_matrix", proj_matrix_inverse)
	_buffer1.shader_material.set_shader_parameter(&"u_inv_view_matrix", view_matrix_inverse)
	_buffer1.shader_material.set_shader_parameter(
		&"u_viewport_size", Vector2(_buffer1.viewport.size))
	
	_buffer0.viewport.render_target_update_mode = SubViewport.UPDATE_DISABLED
	_buffer1.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	
	_texture_rect.texture = _buffer0.viewport.get_texture()
	
	# No idea if this will render before or after we sample it...
	# In fact I'm sure it's already causing problems!!!
#	_render.camera.global_transform = _camera.global_transform
	_render.shader_material.set_shader_parameter(&"u_frame_index", frame_index)
	_render.shader_material.set_shader_parameter(&"u_inv_projection_matrix", proj_matrix_inverse)
	_render.shader_material.set_shader_parameter(&"u_inv_view_matrix", view_matrix_inverse)
	_render.shader_material.set_shader_parameter(&"u_viewport_size", Vector2(_render.viewport.size))
	
	var temp := _buffer0
	_buffer0 = _buffer1
	_buffer1 = temp
	
#	_buffer0.viewport.get_texture().get_image() \
#		.save_png(str("capture/buffer_", str(frame_index).pad_zeros(4), ".png"))
#	_render.viewport.get_texture().get_image() \
#		.save_png(str("capture/render_", str(frame_index).pad_zeros(4), ".png"))
#	get_viewport().get_texture().get_image() \
#		.save_png(str("capture/final_", str(frame_index).pad_zeros(4), ".png"))
	
	_prev_view_proj_matrix = view_proj_matrix
	_prev_camera_transform = _camera.global_transform
	
	var win_size = get_viewport().size
	if _ack_size != win_size:
		_ack_size = win_size
		_resize_targets(win_size)


# Doesn't work??
#func _notification(what):
#	if what == NOTIFICATION_WM_SIZE_CHANGED:
#		var size = get_viewport().size
#		print("Resize ", size)
#		_render.viewport.size = size / 4
#		_buffer0.viewport.size = size
#		_buffer1.viewport.size = size


var _ack_size := Vector2i()
func _resize_targets(size: Vector2i):
#	size /= 2
	# Keep target sizes multiple of downscale factor to keep pixels aligned.
	size = (size / 4) * 4
	print("Resize ", size)
	_render.viewport.size = size / 4
	_buffer0.viewport.size = size
	_buffer1.viewport.size = size
	# TODO For some reason changing viewports size creates huge visual glitches...
#	_buffer0.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
#	_buffer1.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS


