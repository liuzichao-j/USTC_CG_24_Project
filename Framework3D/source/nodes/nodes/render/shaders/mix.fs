#version 430 core

uniform vec2 iResolution;

uniform sampler2D baseColorSampler1;
uniform sampler2D baseColorSampler2;
uniform float alpha;

layout(location = 0) out vec4 Color;


void main() {
	vec2 uv = gl_FragCoord.xy / iResolution;
	Color = mix(texture(baseColorSampler1, uv), texture(baseColorSampler2, uv), alpha);
}
