#include "NODES_FILES_DIR.h"
#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "Nodes/socket_types/basic_socket_types.hpp"
#include "RCore/Backend.hpp"
#include "RCore/ResourceAllocator.hpp"
#include "camera.h"
#include "geometries/mesh.h"
#include "light.h"
#include "material.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hgiGL/computeCmds.h"
#include "render_node_base.h"
#include "resource_allocator_instance.hpp"
#include "rich_type_buffer.hpp"

namespace USTC_CG::node_relativity_rasterize {
static void node_declare(NodeDeclarationBuilder& b)
{
    b.add_input<decl::Camera>("Camera");
    b.add_input<decl::Meshes>("Meshes");
    b.add_input<decl::Materials>("Materials");
    b.add_input<decl::String>("Vertex Shader").default_val("shaders/relativity_rasterize.vs");
    b.add_input<decl::String>("Fragment Shader").default_val("shaders/relativity_rasterize.fs");
    b.add_input<decl::Float>("Speed of Light").default_val(1.0f);
    b.add_input<decl::Float>("Doppler Mix").min(0.0f).max(1.0f).default_val(1.0f);
    b.add_input<decl::Int>("Draw Grid").min(0).max(1).default_val(1);
    b.add_output<decl::Texture>("Position");
    b.add_output<decl::Texture>("Depth");
    b.add_output<decl::Texture>("Texcoords");
    b.add_output<decl::Texture>("diffuseColor");
    b.add_output<decl::Texture>("MetallicRoughness");
    b.add_output<decl::Texture>("Normal");
}

static void node_exec(ExeParams params)
{
    auto cameras = params.get_input<CameraArray>("Camera");
    auto meshes = params.get_input<MeshArray>("Meshes");
    MaterialMap materials = params.get_input<MaterialMap>("Materials");
    auto speed_of_light = params.get_input<float>("Speed of Light");
    auto doppler_mix = params.get_input<float>("Doppler Mix");
    int draw_grid = params.get_input<int>("Draw Grid");

    Hd_USTC_CG_Camera* free_camera;

    for (auto camera : cameras) {
        if (camera->GetId() != SdfPath::EmptyPath()) {
            free_camera = camera;
            break;
        }
    }

    auto size = free_camera->_dataWindow.GetSize();

    TextureDesc texture_desc;
    texture_desc.size = size;
    texture_desc.format = HdFormatFloat32Vec3;
    auto position_texture = resource_allocator.create(texture_desc);
    auto normal_texture = resource_allocator.create(texture_desc);

    texture_desc.format = HdFormatFloat32UInt8;
    auto depth_texture_for_opengl = resource_allocator.create(texture_desc);

    texture_desc.format = HdFormatFloat32;
    auto depth_texture = resource_allocator.create(texture_desc);

    texture_desc.format = HdFormatFloat32Vec2;
    auto texcoords_texture = resource_allocator.create(texture_desc);

    texture_desc.format = HdFormatFloat32Vec2;
    auto metallic_roughness = resource_allocator.create(texture_desc);

    texture_desc.format = HdFormatFloat32Vec3;
    auto diffuseColor_texture = resource_allocator.create(texture_desc);

    auto vs_path = params.get_input<std::string>("Vertex Shader");
    auto fs_path = params.get_input<std::string>("Fragment Shader");

    ShaderDesc shader_desc;
    shader_desc.set_vertex_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) / std::filesystem::path(vs_path));

    shader_desc.set_fragment_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) / std::filesystem::path(fs_path));
    auto shader_handle = resource_allocator.create(shader_desc);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, position_texture->texture_id, 0);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, depth_texture->texture_id, 0);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, texcoords_texture->texture_id, 0);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, diffuseColor_texture->texture_id, 0);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, metallic_roughness->texture_id, 0);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_TEXTURE_2D, normal_texture->texture_id, 0);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_DEPTH_STENCIL_ATTACHMENT,
        GL_TEXTURE_2D,
        depth_texture_for_opengl->texture_id,
        0);

    GLenum attachments[6] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
                              GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5 };
    glDrawBuffers(6, attachments);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    glViewport(0, 0, size[0], size[1]);

    shader_handle->shader.use();
    shader_handle->shader.setMat4("view", GfMatrix4f(free_camera->_viewMatrix));
    shader_handle->shader.setMat4("projection", GfMatrix4f(free_camera->_projMatrix));

    auto camera_mat = free_camera->GetTransform();
    GfVec3f camera_pos = { (float)camera_mat[3][0],
                           (float)camera_mat[3][1],
                           (float)camera_mat[3][2] };

    shader_handle->shader.setVec3("camPos", camera_pos);
	shader_handle->shader.setFloat("dopplerMix", doppler_mix);

    if (draw_grid) 
    {
        shader_handle->shader.setVec3("camSpeed", GfVec3f(0.0, 0.0, 0.0));
        shader_handle->shader.setFloat("lightSpeed", 10000000);
        glColor4f(0.0, 0.0, 0.0, 0.2);
        glBegin(GL_LINES);
        const int gridn = 20;
        const float gridy = -0.01;
        for (int i = -gridn; i <= gridn; ++i) {
            glVertex3f(-gridn, i, gridy);
            glVertex3f(gridn, i, gridy);
            glVertex3f(i, -gridn, gridy);
            glVertex3f(i, gridn, gridy);
        }
        glEnd();
    }

    shader_handle->shader.setVec3("camSpeed", free_camera->_velocity);
    shader_handle->shader.setFloat("lightSpeed", speed_of_light);
    // std::cout << "speed: " << free_camera->_velocity << "; place: " << camera_pos << "; light: " << speed_of_light << std::endl;

    for (int i = 0; i < meshes.size(); ++i) {
        auto mesh = meshes[i];

        shader_handle->shader.setMat4("model", mesh->transform);
        auto material = materials[mesh->GetMaterialId()];

        material->RefreshGLBuffer();
        material->BindTextures(shader_handle->shader);

        TfToken texcoordName = material->requireTexcoordName();

        mesh->RefreshGLBuffer();
        mesh->RefreshTexcoordGLBuffer(texcoordName);

        glBindVertexArray(mesh->VAO);
        glDeleteBuffers(1, &mesh->velocityBuffer);
        glGenBuffers(1, &mesh->velocityBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, mesh->velocityBuffer);
        glBufferData(
            GL_ARRAY_BUFFER,
            mesh->vertex_velocity.size() * sizeof(GfVec3f),
            mesh->vertex_velocity.cdata(),
            GL_DYNAMIC_DRAW);

        glVertexAttribPointer(mesh->velocityLocation, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
        glEnableVertexAttribArray(mesh->velocityLocation);

        glBindVertexArray(mesh->VAO);
        glDrawElements(
            GL_TRIANGLES,
            static_cast<unsigned int>(mesh->triangulatedIndices.size() * 3),
            GL_UNSIGNED_INT,
            0);
        glBindVertexArray(0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &framebuffer);

    auto shader_error = shader_handle->shader.get_error();

    resource_allocator.destroy(shader_handle);
    resource_allocator.destroy(depth_texture_for_opengl);

    params.set_output("Position", position_texture);
    params.set_output("Normal", normal_texture);
    params.set_output("Depth", depth_texture);
    params.set_output("Texcoords", texcoords_texture);
    params.set_output("MetallicRoughness", metallic_roughness);
    params.set_output("diffuseColor", diffuseColor_texture);

    if (!shader_error.empty()) {
        throw std::runtime_error(shader_error);
    }
}

static void node_register()
{
    static NodeTypeInfo ntype;

    strcpy(ntype.ui_name, "Relativity Rasterize");
    strcpy_s(ntype.id_name, "render_relativity_rasterize");

    render_node_type_base(&ntype);
    ntype.node_execute = node_exec;
    ntype.declare = node_declare;

    nodeRegisterType(&ntype);
}

NOD_REGISTER_NODE(node_register)
}  // namespace USTC_CG::node_relativity_rasterize