#include "NODES_FILES_DIR.h"
#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "Nodes/socket_types/basic_socket_types.hpp"
#include "camera.h"
#include "light.h"
#include "pxr/imaging/hd/tokens.h"
#include "render_node_base.h"
#include "resource_allocator_instance.hpp"
#include "rich_type_buffer.hpp"
#include "utils/draw_fullscreen.h"

namespace USTC_CG::node_mix_color {
static void node_declare(NodeDeclarationBuilder& b)
{
    b.add_input<decl::Texture>("Color1");
    b.add_input<decl::Texture>("Color2");
    b.add_input<decl::Float>("Alpha").default_val(1.0f).min(0.0f).max(1.0f);

    b.add_input<decl::String>("Shader").default_val("shaders/mix.fs");
    b.add_output<decl::Texture>("Color");
}

static void node_exec(ExeParams params)
{
    auto color1 = params.get_input<TextureHandle>("Color1");
    auto color2 = params.get_input<TextureHandle>("Color2");
    auto alpha = params.get_input<float>("Alpha");

    auto size = color1->desc.size;

    unsigned int VBO, VAO;

    CreateFullScreenVAO(VAO, VBO);

    TextureDesc texture_desc;
    texture_desc.size = size;
    texture_desc.format = HdFormatFloat32Vec4;
    auto color_texture = resource_allocator.create(texture_desc);

    auto shaderPath = params.get_input<std::string>("Shader");

    ShaderDesc shader_desc;
    shader_desc.set_vertex_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) /
        std::filesystem::path("shaders/fullscreen.vs"));

    shader_desc.set_fragment_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) / std::filesystem::path(shaderPath));
    auto shader = resource_allocator.create(shader_desc);
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture->texture_id, 0);

    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader->shader.use();
    shader->shader.setVec2("iResolution", size);

    shader->shader.setInt("baseColorSampler1", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, color1->texture_id);

    shader->shader.setInt("baseColorSampler2", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, color2->texture_id);

    shader->shader.setFloat("alpha", alpha);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    DestroyFullScreenVAO(VAO, VBO);
    resource_allocator.destroy(shader);

    params.set_output("Color", color_texture);
}

static void node_register()
{
    static NodeTypeInfo ntype;

    strcpy(ntype.ui_name, "Mix Color");
    strcpy_s(ntype.id_name, "render_mix_color");

    render_node_type_base(&ntype);
    ntype.node_execute = node_exec;
    ntype.declare = node_declare;
    nodeRegisterType(&ntype);
}

NOD_REGISTER_NODE(node_register)
}  // namespace USTC_CG::node_mix_color
