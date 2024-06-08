#version 430 core

uniform vec2 iResolution;

uniform sampler2D positionSampler;
uniform sampler2D normalSampler;
uniform sampler2D depthSampler;
uniform sampler2D baseColorSampler;

uniform mat4 projection;
uniform mat4 view;

layout(location = 0) out vec4 Color;

// Constants
const float PI = 3.14159265359;
const float TWO_PI = 6.2831853071;

vec3 bg_color(float t)
{
	return mix(vec3(0.04), vec3(0.8, 0.8, 1.0), t * t);
}

void main() {
	vec2 uv = gl_FragCoord.xy / iResolution;
	float depth_val = texture2D(depthSampler, uv).x;

	vec4 clipSpacePos = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
	mat4 view_mat = mat4(mat3(view));

	vec4 worldSpacePos = inverse(projection * view_mat) * clipSpacePos;
	vec3 position = worldSpacePos.xyz;
	vec3 dir = normalize(position);
	vec3 env_color = bg_color(dir.z * 0.5 + 0.5);

	if(depth_val == 0) {
		Color = vec4(env_color, 1.0);
	} 
	else 
	{
		float dist = length(texture(positionSampler, uv).xyz - position);
		float decay = 1.0, decaydist = 10.0;
		if(dist > decaydist) decay = pow(1.4, -dist + decaydist);
		vec3 color_val = texture(baseColorSampler, uv).xyz;
		Color = vec4(mix(env_color, color_val, decay), 1.0);
	}
}
