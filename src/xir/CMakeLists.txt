set(LUISA_COMPUTE_XIR_SOURCES
        argument.cpp
        basic_block.cpp
        builder.cpp
        constant.cpp
        debug_printer.cpp
        function.cpp
        instruction.cpp
        metadata.cpp
        module.cpp
        op.cpp
        special_register.cpp
        use.cpp
        user.cpp
        value.cpp
        verifier.cpp
        verifier_dom_tree.cpp

        # derived instructions
        instructions/alloca.cpp
        instructions/arithmetic.cpp
        instructions/assert.cpp
        instructions/assume.cpp
        instructions/atomic.cpp
        instructions/autodiff.cpp
        instructions/branch.cpp
        instructions/break.cpp
        instructions/call.cpp
        instructions/cast.cpp
        instructions/coro.cpp
        instructions/clock.cpp
        instructions/continue.cpp
        instructions/debug_break.cpp
        instructions/gep.cpp
        instructions/if.cpp
        instructions/indexed_branch.cpp
        instructions/load.cpp
        instructions/loop.cpp
        instructions/outline.cpp
        instructions/phi.cpp
        instructions/print.cpp
        instructions/raster_discard.cpp
        instructions/ray_query.cpp
        instructions/resource.cpp
        instructions/return.cpp
        instructions/store.cpp
        instructions/switch.cpp
        instructions/thread_group.cpp
        instructions/unreachable.cpp

        # derived metadata
        metadata/location.cpp
        metadata/comment.cpp
        metadata/name.cpp
        metadata/curve_basis.cpp
        metadata/signature_constraint.cpp
        metadata/reg2mem_spill.cpp

        # translators
        translators/ast2xir.cpp
        translators/json2xir.cpp
        translators/xir_interchange.cpp
        translators/xir2ast.cpp
        translators/xir2json.cpp
        translators/xir2text.cpp

        # passes
        passes/early_cse.cpp
        passes/helpers.cpp
        passes/dce.cpp
        passes/uniformity_analysis.cpp
        passes/dom_tree.cpp
        passes/call_graph.cpp
        passes/convergence_region.cpp
        passes/coro_cfg_distill.cpp
        passes/coro_materialize.cpp
        passes/dead_field_elimination.cpp
        passes/coro_reg2mem.cpp
        passes/coro_split.cpp
        passes/outline.cpp
        passes/dead_store_elimination.cpp
        passes/sroa.cpp
        passes/trace_gep.cpp
        passes/aggregate_field_bitmask.cpp
        passes/early_return_elimination.cpp
        passes/fix_self_referential.cpp
        passes/pointer_usage.cpp
        passes/post_dom_tree.cpp
        passes/lex_scope_analysis.cpp
        passes/licm.cpp
        passes/local_store_forward.cpp
        passes/local_load_elimination.cpp
        passes/lower_ray_query_loop.cpp
        passes/lower_ray_query_loop_to_loop.cpp
        passes/mem2reg.cpp
        passes/autodiff.cpp
        passes/transpose_gep.cpp
        passes/unused_callable_removal.cpp
        passes/reg2mem.cpp
        passes/promote_ref_arg.cpp
        passes/reassociate.cpp
        passes/algebraic_simplify.cpp
        passes/alias_analysis.cpp
        passes/const_fold.cpp
        passes/dead_arg_elim.cpp
        passes/cvp.cpp
        passes/sccp.cpp
        passes/scalarizer.cpp
        passes/inline.cpp
        passes/natural_loop.cpp
        passes/loop_fusion.cpp
        passes/loop_rotation.cpp
        passes/loop_vectorization.cpp
        passes/slp_vectorization.cpp
        passes/lower_break_continue.cpp
        passes/destructure_cfg.cpp
        passes/div_rem_pairs.cpp
        passes/simplify_cfg.cpp
        passes/simplify_libcalls.cpp
        passes/restructure_cfg.cpp
        passes/restructure_cfg_selection_merge.cpp
        passes/scalar_evolution.cpp
        passes/gvn.cpp
        passes/phi_cleanup.cpp
        passes/if_conversion.cpp
        passes/indvar_simplify.cpp
        passes/pass_pipeline.cpp
        passes/fuse_consecutive_buffer_reads.cpp
)

add_library(luisa-compute-xir SHARED ${LUISA_COMPUTE_XIR_SOURCES})
target_link_libraries(luisa-compute-xir PUBLIC luisa-compute-ast)

if (NOT LUISA_COMPUTE_USE_SYSTEM_YYJSON)
    target_link_libraries(luisa-compute-xir PRIVATE luisa-compute-ext-yyjson-interface)
endif ()

target_compile_definitions(luisa-compute-xir PUBLIC LUISA_ENABLE_XIR=1)
target_compile_definitions(luisa-compute-xir PRIVATE LUISA_XIR_EXPORT_DLL=1)

set_target_properties(luisa-compute-xir PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME luisa-xir)

luisa_compute_install(xir)

if (LUISA_COMPUTE_ENABLE_XIR_TESTS)

    function(luisa_compute_xir_add_test name)
        add_executable(xir_test_${name} tests/test_${name}.cpp)
        target_link_libraries(xir_test_${name} PRIVATE luisa::compute)
    endfunction()

    luisa_compute_xir_add_test(aggregate_field_bitmasks)
    luisa_compute_xir_add_test(mem2reg)
    luisa_compute_xir_add_test(debug_printer)
    luisa_compute_xir_add_test(dce)
    luisa_compute_xir_add_test(ray_query)

endif ()
