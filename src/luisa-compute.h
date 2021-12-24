//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <core/stl.h>
#include <core/atomic.h>
#include <core/basic_traits.h>
#include <core/basic_types.h>
#include <core/clock.h>
#include <core/concepts.h>
#include <core/constants.h>
#include <core/dirty_range.h>
#include <core/dynamic_module.h>
#include <core/first_fit.h>
#include <core/hash.h>
#include <core/intrin.h>
#include <core/logging.h>
#include <core/lru_cache.h>
#include <core/macro.h>
#include <core/mathematics.h>
#include <core/observer.h>
#include <core/platform.h>
#include <core/pool.h>
#include <core/rc.h>
#include <core/spin_mutex.h>
#include <core/thread_pool.h>

#include <ast/constant_data.h>
#include <ast/expression.h>
#include <ast/function.h>
#include <ast/function_builder.h>
#include <ast/interface.h>
#include <ast/op.h>
#include <ast/statement.h>
#include <ast/type.h>
#include <ast/type_registry.h>
#include <ast/usage.h>
#include <ast/variable.h>

#include <runtime/bindless_array.h>
#include <runtime/buffer.h>
#include <runtime/command.h>
#include <runtime/command_buffer.h>
#include <runtime/command_list.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/event.h>
#include <runtime/image.h>
#include <runtime/mipmap.h>
#include <runtime/pixel.h>
#include <runtime/resource.h>
#include <runtime/resource_tracker.h>
#include <runtime/sampler.h>
#include <runtime/shader.h>
#include <runtime/stream.h>
#include <runtime/volume.h>

#include <dsl/arg.h>
#include <dsl/builtin.h>
#include <dsl/constant.h>
#include <dsl/expr.h>
#include <dsl/expr_traits.h>
#include <dsl/func.h>
#include <dsl/operators.h>
#include <dsl/ref.h>
#include <dsl/shared.h>
#include <dsl/stmt.h>
#include <dsl/struct.h>
#include <dsl/sugar.h>
#include <dsl/syntax.h>
#include <dsl/var.h>

#include <rtx/accel.h>
#include <rtx/hit.h>
#include <rtx/mesh.h>
#include <rtx/ray.h>

#include <gui/framerate.h>
#include <gui/window.h>
