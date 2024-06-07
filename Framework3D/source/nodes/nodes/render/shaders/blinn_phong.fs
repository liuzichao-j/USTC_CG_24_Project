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

// https://stackoverflow.com/questions/9446888/best-way-to-detect-nans-in-opengl-shaderms
bool isnan( float val )
{
  return ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
  // important: some nVidias failed to cope with version below.
  // Probably wrong optimization.
  /*return ( val <= 0.0 || 0.0 <= val ) ? false : true;*/
}

void main() 
{
    vec2 uv = gl_FragCoord.xy / iResolution;
    vec3 frag_pos = texture(position, uv).xyz;

    vec4 metalnessRoughness = texture(metallicRoughnessSampler,uv);
    float metal = metalnessRoughness.x;
    float roughness = metalnessRoughness.y;

    vec3 norm = normalize(texture(normalMapSampler, uv).xyz);

    if(isnan(norm.x))
    {
        //background
        Color = vec4(vec3(0.05), 1.0);
        return;
    }

    vec3 diff_color = texture(diffuseColorSampler, uv).rgb;
    vec3 result = vec3(0.0);

    //https://kcoley.github.io/glTF/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows/
    const vec3 dielectricSpecular = vec3(0.04, 0.04, 0.04);
    const float epsilon = 1e-6;
    vec3 specular = mix(dielectricSpecular, diff_color, metal);
    vec3 oneMinusSpecularStrength = 1.0 - specular; 
    vec3 diffuse = diff_color * (1 - dielectricSpecular) * (1 - metal) / max(oneMinusSpecularStrength, epsilon);
    float glossiness = 1 - roughness;

    for(int i = 0; i < light_count; i++) 
    {
        float dist_sq = dot(frag_pos - lights[i].position, frag_pos - lights[i].position);

        vec3 lightDir = normalize(lights[i].position - frag_pos);
        vec3 viewDir = normalize(iCameraPos - frag_pos);
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float NdotL = max(dot(norm, lightDir), 0.0);
        float NdotH = max(dot(norm, halfwayDir), 0.0);
        
        vec3 ambient = lights[i].color * 0.1 * diff_color;
        vec3 diff = lights[i].color * NdotL * diffuse;
        vec3 spec = lights[i].color * specular * pow(NdotH, glossiness * 31.0 + 1.0);

        result += (ambient + diff + spec) / dist_sq;

    }
    Color = vec4(result, 1.0);
    
    // Gamma correction
    float gamma = 2.2;
    Color.rgb = pow(Color.rgb, vec3(1.0 / gamma));
}