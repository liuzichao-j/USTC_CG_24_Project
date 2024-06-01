#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(std430, binding = 0) buffer buffer0 {
vec2 data[];
}
aTexcoord;

out vec3 vertexPosition;
out vec3 vertexNormal;
out vec2 vTexcoord;

uniform float lightSpeed;
uniform vec3 camPos;
uniform vec3 camSpeed;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
vec4 vPosition = model * vec4(aPos, 1.0);
vertexPosition = vPosition.xyz / vPosition.w;

// Constants
vec3 beta = camSpeed / lightSpeed;
float gamma = 1 / sqrt(1 - dot(beta, beta));

// Get the vector from the camera to the vertex
vec3 dir = vertexPosition - camPos;
if(length(beta) > 0)
{
float dist = length(dir);

dir = dir + gamma*gamma/(gamma+1) * dot(beta, dir) * beta - gamma * dist * beta;

//dir = normalize(dir);
//float paradist = gamma * (dot(beta, dir) / length(beta) - length(beta) * length(dir));
// Change the direction
//dir = normalize(dir - gamma * beta + (gamma - 1) * beta * dot(beta, dir) / dot(beta, beta));
// Get the new position
//dir = dir * paradist / dot(beta, dir) * length(beta);
}
vertexPosition = camPos + dir;

vPosition = vec4(vertexPosition * vPosition.w, vPosition.w);

gl_Position = projection * view * vPosition;
vertexNormal = (inverse(transpose(mat3(model))) * aNormal);
vTexcoord = aTexcoord.data[gl_VertexID];
vTexcoord.y = 1.0 - vTexcoord.y;
}