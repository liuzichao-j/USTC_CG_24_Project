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

uniform int shadow_map_resolution;

// uniform float alpha;

uniform int light_count;

layout(location = 0) out vec4 Color;

float PCSS(vec3 frag_pos, int light_index, float NdotL)
{
    vec4 frag_pos_light_space = lights[light_index].light_projection * lights[light_index].light_view * vec4(frag_pos, 1.0); 
    const int sample_range_blocker = 5, sample_range_shadow = 9;
    float sample_r = 1.0 / shadow_map_resolution;
    float blocker = 0.0;
    int blocker_n = 0;

    float bias = max(0.05 * (1.0 - NdotL), 0.005);
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    float current_depth = frag_pos_light_space.z / frag_pos_light_space.w;

    for(int i = -sample_range_blocker; i <= sample_range_blocker; i++)
    {
        for(int j = -sample_range_blocker; j <= sample_range_blocker; j++)
        {
            vec2 offset = vec2(i, j) * sample_r;
            float sample_depth = texture(shadow_maps, vec3(proj_coords.xy + offset, lights[light_index].shadow_map_id)).r;
            // 加上 bias 以避免自遮挡
            if(current_depth + bias > sample_depth)
            {
                blocker += sample_depth;
                blocker_n++;
            }
        }
    }

    blocker /= float(blocker_n);

    float w = (current_depth - blocker) / blocker * lights[light_index].radius;

    float shadow_sum = 0.0;
    
    // Maybe not correct
    for(int i = -sample_range_shadow; i <= sample_range_shadow; i++)
    {
        for(int j = -sample_range_shadow; j <= sample_range_shadow; j++)
        {
            vec2 offset = vec2(i, j) / float(sample_range_shadow) * w;
            float sample_depth = texture(shadow_maps, vec3(proj_coords.xy + offset, lights[light_index].shadow_map_id)).r;
            if(current_depth - bias > sample_depth)
            {
                shadow_sum += 1.0;
            }
        }
    }

    return shadow_sum / (sample_range_shadow * 2 + 1) / (sample_range_shadow * 2 + 1);
}

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
        float shadow_map_value = texture(shadow_maps, vec3(uv, lights[i].shadow_map_id)).x;
        float dist_sq = dot(frag_pos - lights[i].position, frag_pos - lights[i].position);

        vec3 lightDir = normalize(lights[i].position - frag_pos);
        vec3 viewDir = normalize(iCameraPos - frag_pos);
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float NdotL = max(dot(norm, lightDir), 0.0);
        float NdotH = max(dot(norm, halfwayDir), 0.0);
        
        vec3 ambient = lights[i].color * 0.05 * diff_color;
        vec3 diff = lights[i].color * NdotL * diffuse;
        vec3 spec = lights[i].color * specular * pow(NdotH, glossiness * 31.0 + 1.0);

        float shadow = PCSS(frag_pos, i, NdotL);
        result += (ambient + (1.0 - shadow) * (diff + spec)) / dist_sq;
    }
    Color = vec4(result, 1.0);
    
    // Gamma correction
    float gamma = 2.2;
    Color.rgb = pow(Color.rgb, vec3(1.0 / gamma));
}