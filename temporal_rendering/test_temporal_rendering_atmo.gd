extends Node

const PlanetAtmosphere = preload("res://addons/zylann.atmosphere/planet_atmosphere.gd")
const TemporalRenderer = preload("./temporal_renderer.gd")

const PlanetAtmosphereCloudsTemporalShader = \
	preload("res://addons/zylann.atmosphere/shaders/planet_atmosphere_clouds_temporal.gdshader")

@onready var _atmosphere : PlanetAtmosphere = $Spatial/PlanetAthmosphere
@onready var _avatar : Node3D = $Spatial/Avatar


func _ready():
	_atmosphere.clouds_rotation_speed = 0.05
#	_atmosphere.set_shader_parameter(&"u_cloud_detail_animation_speed", 0.0)
	_atmosphere.set_shader_parameter(&"u_cloud_detail_scale", 0.5)
#	_atmosphere.set_shader_parameter(&"u_cloud_density_scale", 2.0)
	_atmosphere.set_shader_parameter(&"u_cloud_blend", 0.2)
#	_atmosphere.set_shader_parameter(&"u_cloud_bottom", 0.0)
#	_atmosphere.set_shader_parameter(&"u_cloud_top", 0.1)
#	_atmosphere.set_shader_parameter(&"u_sphere_depth_factor", 1.0)
	_atmosphere.custom_shader = PlanetAtmosphereCloudsTemporalShader
	var temporal_renderer := TemporalRenderer.new()
	_atmosphere.add_child(temporal_renderer)
	_avatar.speed = 1.0


func _input(event):
	if event is InputEventKey:
		if event.pressed:
			if event.keycode == KEY_KP_4:
				var right := get_viewport().get_camera_3d().global_transform.basis.x
				_avatar.global_position -= right * 10.0
			elif event.keycode == KEY_KP_5:
				var right := get_viewport().get_camera_3d().global_transform.basis.x
				_avatar.global_position += right * 10.0


func _process(delta):
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		_atmosphere.set_shader_parameter(&"u_sphere_depth_factor", 1.0)
	else:
		_atmosphere.set_shader_parameter(&"u_sphere_depth_factor", 0.0)
#	var im = get_viewport().get_texture().get_image()
#	im.save_png(str("tests/debug_data/screen_frame_", 
#		str(Engine.get_frames_drawn()).pad_zeros(4), ".png"))

