#version 330 core

uniform vec3 Color;
uniform vec3 Light;
uniform bool use_lighting;

in vec3 v_norm;

out vec4 f_color;
void main() {
	if (!use_lighting) {
		// No lighting, but should still use v_norm against the camera
		f_color = vec4(Color, 1.0);
		return;
	}
	vec3 l = normalize(Light);
	vec3 n = normalize(v_norm);
	float lum = max(abs(dot(n,l)), 0.1) + 0.2;
	f_color = vec4(Color * lum, 1.0);
}