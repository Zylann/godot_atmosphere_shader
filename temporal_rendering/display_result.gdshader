shader_type canvas_item;

void fragment() {
	vec4 col = texture(TEXTURE, vec2(UV.x, UV.y));
	COLOR = vec4(0.0, col.b, 0.0, 1.0);
}

