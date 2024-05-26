#version 430 core

// Define a uniform struct for lights
struct Light {
    // The matrices are used for shadow mapping. You need to fill it according to how we are filling it when building the normal maps (node_render_shadow_mapping.cpp). 
    // Now, they are filled with identity matrix. You need to modify C++ code innode_render_deferred_lighting.cpp.
    // Position and color are filled.
    mat4 light_projection;
    mat4 light_view;
    vec3 position;
    float radius;
    vec3 color; // Just use the same diffuse and specular color.
    int shadow_map_id;
};

layout(binding = 0) buffer lightsBuffer {
Light lights[4];
};

uniform vec2 iResolution;

uniform vec3 iCameraPos;

uniform sampler2D diffuseColorSampler;
uniform sampler2D normalMapSampler; // You should apply normal mapping in rasterize_impl.fs
uniform sampler2D metallicRoughnessSampler;
uniform sampler2DArray shadow_maps;
uniform sampler2D position;

// uniform float alpha;
uniform vec3 camPos;

uniform int light_count;

layout(location = 0) out vec4 Color;

const float PI = 3.14159265359;

// https://stackoverflow.com/questions/9446888/best-way-to-detect-nans-in-opengl-shaderms
bool isnan( float val )
{
  return ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
  // important: some nVidias failed to cope with version below.
  // Probably wrong optimization.
  /*return ( val <= 0.0 || 0.0 <= val ) ? false : true;*/
}

// Metalness-Roughness workflow
// https://zhuanlan.zhihu.com/p/304191958
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
float DistributionGGX(float NdotH, float a)
{
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;
    return step(NdotH, 0) * a2 / denom;
}
float SchlickGGX(float cosTheta, float k)
{
    return cosTheta / (cosTheta * (1 - k) + k);
}

void main() 
{
    vec2 uv = gl_FragCoord.xy / iResolution;
    vec3 frag_pos = texture(position, uv).xyz;

    vec4 metalnessRoughness = texture(metallicRoughnessSampler,uv);
    float metal = metalnessRoughness.x;
    float roughness = metalnessRoughness.y;
    float rough_sq = roughness * roughness;

    vec3 norm = normalize(texture(normalMapSampler, uv).xyz);
    if(isnan(norm.x))
    {
        //background
        Color = vec4(vec3(0.1), 1.0);
        return;
    }

    vec3 diff_color = texture(diffuseColorSampler, uv).rgb;
    vec3 result = vec3(0.0);

    for(int i = 0; i < light_count; i++) 
    {
        float shadow_map_value = texture(shadow_maps, vec3(uv, lights[i].shadow_map_id)).x;

        float dist_sq = dot(frag_pos - lights[i].position, frag_pos - lights[i].position);

        vec3 lightDir = normalize(lights[i].position - frag_pos);
        vec3 viewDir = normalize(iCameraPos - frag_pos);
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float VdotH = max(dot(viewDir, halfwayDir), 0.0);
        float NdotH = max(dot(norm, halfwayDir), 0.0);      // Blinn-Phong 高光
        float NdotL = max(dot(norm, lightDir), 0.0);        // 漫反射强度
        float NdotV = max(dot(norm, viewDir), 0.0);

        vec3 F0 = vec3(0.04);
        F0 = mix(F0, diff_color, metal);
        vec3 F = fresnelSchlick(VdotH, F0);
        vec3 kd = (1 - F) * (1 - metal);

        float D = DistributionGGX(NdotH, rough_sq);

        float k_dir = pow(rough_sq + 1, 2) / 8;
        float ggx1 = SchlickGGX(NdotL, k_dir);
        float ggx2 = SchlickGGX(NdotV, k_dir);
        float G = ggx1 * ggx2;
        vec3 FDG = F * D * G;

        vec3 ambient = vec3(0.1);
        vec3 diffuse = diff_color * kd / PI;
        vec3 spec = FDG / max(4 * NdotL * NdotV, 0.001);

        result += (1 - metal) * diff_color * ambient / PI / dist_sq;
        result += (diffuse + spec) * lights[i].color * NdotL / dist_sq;
    }
    Color = vec4(result, 1.0);
    
    // Gamma correction
    float gamma = 2.2;
    Color.rgb = pow(Color.rgb, vec3(1.0 / gamma));
}