#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "geom_node_base.h"

namespace USTC_CG::node_speed_of_light {
static void node_declare_set(NodeDeclarationBuilder& b)
{
    b.add_input<decl::Float>("speed").min(0.0).max(1000000.0).default_val(1000.0);
}

static void node_exec_set(ExeParams params)
{
    // This is for external read. Do nothing.
}

static void node_declare_get(NodeDeclarationBuilder& b)
{
    b.add_output<decl::Float>("speed");
}

static void node_exec_get(ExeParams params)
{
    // This is for external write. Do nothing.
}

static void node_register()
{
    static NodeTypeInfo ntype_set;
    strcpy(ntype_set.ui_name, "Set Speed of Light");
    strcpy_s(ntype_set.id_name, "geom_speed_of_light_set");
    geo_node_type_base(&ntype_set);
    ntype_set.node_execute = node_exec_set;
    ntype_set.declare = node_declare_set;
    ntype_set.ALWAYS_REQUIRED = true;
    nodeRegisterType(&ntype_set);

    static NodeTypeInfo ntype_get;
    strcpy(ntype_get.ui_name, "Get Speed of Light");
    strcpy_s(ntype_get.id_name, "geom_speed_of_light_get");
    geo_node_type_base(&ntype_get);
    ntype_get.node_execute = node_exec_get;
    ntype_get.declare = node_declare_get;
    ntype_get.ALWAYS_REQUIRED = true;
    nodeRegisterType(&ntype_get);
}

NOD_REGISTER_NODE(node_register)
}  // namespace USTC_CG::node_speed_of_light