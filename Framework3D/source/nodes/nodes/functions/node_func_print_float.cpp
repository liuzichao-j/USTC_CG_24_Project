#include "Nodes/node.hpp"
#include "Nodes/node_declare.hpp"
#include "Nodes/node_register.h"
#include "func_node_base.h"

namespace USTC_CG::node_print_float {
static void node_declare(NodeDeclarationBuilder& b)
{
    b.add_input<decl::Float>("Float").default_val(0.1f).min(0.0f).max(1.0f);
}

static void node_exec(ExeParams params)
{
    auto anyfloat = params.get_input<float>("Float");

    std::cout << "Float: " << anyfloat << std::endl;
}

static void node_register()
{
    static NodeTypeInfo ntype;

    strcpy(ntype.ui_name, "Print Float");
    strcpy_s(ntype.id_name, "render_print_float");

    func_node_type_base(&ntype);
    ntype.node_execute = node_exec;
    ntype.ALWAYS_REQUIRED = true;
    ntype.declare = node_declare;
    nodeRegisterType(&ntype);
}

NOD_REGISTER_NODE(node_register)
}  // namespace USTC_CG::node_print_float