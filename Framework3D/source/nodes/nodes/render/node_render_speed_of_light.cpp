#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "camera.h"
#include "geometries/mesh.h"
#include "light.h"
#include "material.h"
#include "pxr/imaging/hd/tokens.h"
#include "render_node_base.h"
#include "rich_type_buffer.hpp"
#include "Nodes/socket_types/basic_socket_types.hpp"

namespace USTC_CG::node_render_speed_of_light {
static void node_declare(NodeDeclarationBuilder& b)
{
    b.add_output<decl::Float>("speed");
}

static void node_exec(ExeParams params)
{
    // This is for external write. Do nothing.
}

static void node_register()
{
    static NodeTypeInfo ntype;
    strcpy(ntype.ui_name, "Speed of Light");
    strcpy_s(ntype.id_name, "render_speed_of_light");
    render_node_type_base(&ntype);
    ntype.node_execute = node_exec;
    ntype.declare = node_declare;
    nodeRegisterType(&ntype);
}

NOD_REGISTER_NODE(node_register)
}  // namespace USTC_CG::node_render_speed_of_light