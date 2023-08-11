#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>
#include <luisa/core/logging.h>
#include <luisa/tensor/tensor.h>

using namespace luisa;
using namespace luisa::compute;

void test_blas_level_1(tensor::TensorMaker &maker, tensor::LASInterface *las, Stream &stream) {
    LUISA_INFO("====================test blas level 1====================");
    auto ID = 0;
    {// norm 2

        // host
        luisa::vector<float> h_x = {5.0f, 12.0f};
        float h_r;

        // device
        auto r = maker.scalar<float>();
        auto x = maker.dense_vector(h_x.size());

        stream << x.dense_storage().copy_from(h_x.data());
        las->nrm2(r, x);
        stream << r.dense_storage().copy_to(&h_r);
        stream << synchronize();
        LUISA_INFO("{}: norm2([5,12])={} (13)", ID++, h_r);
    }

    {// dot
        // host
        luisa::vector<float> h_x = {1.0f, 2.0f, 3.0f};
        luisa::vector<float> h_y = {4.0f, 5.0f, 6.0f};
        float h_r;

        // device
        auto r = maker.scalar<float>();
        auto x = maker.dense_vector(h_x.size());
        auto y = maker.dense_vector(h_y.size());

        stream << x.dense_storage().copy_from(h_x.data());
        stream << y.dense_storage().copy_from(h_y.data());
        las->dot(r, x, y);
        stream << r.dense_storage().copy_to(&h_r);
        stream << synchronize();
        LUISA_INFO("{}: dot([1,2,3],[4,5,6])={} (32)", ID++, h_r);
    }

    {// Iamax & Iamin

        // host
        luisa::vector<float> h_x = {2.0f, 4.0f, 6.0f};
        int h_r = 3;

        // device
        auto r = maker.scalar<int>();
        auto x = maker.dense_vector(h_x.size());
        auto nhandle = (uint64_t)r.dense_storage().buffer.native_handle();
        stream << x.dense_storage().copy_from(h_x.data()) << r.dense_storage().copy_from(&h_r);
        las->iamax(r, x);
        stream << r.dense_storage().copy_to(&h_r);
        stream << synchronize();
        LUISA_INFO("{}: iamax([2,4,6])={} (3)", ID++, h_r);

        las->iamin(r, x);
        stream << r.dense_storage().copy_to(&h_r);
        stream << synchronize();
        LUISA_INFO("{}: iamin([2,4,6])={} (1)", ID++, h_r);
    }
}

void test_blas_level_2(tensor::TensorMaker &maker, tensor::LASInterface *las, Stream &stream) {
    LUISA_INFO("====================test blas level 2====================");
    auto ID = 0;
    {// mv y = alpha * A * x + beta * y
        luisa::vector<float> h_A = {
            1.0f, 3.0f,// col 0
            2.0f, 4.0f // col 1
        };

        luisa::vector<float> h_x = {1.0f, 0.0f};
        luisa::vector<float> h_y = {0.0f, 0.0f};
        float h_alpha = 1.0f;
        float h_beta = 0.0f;

        auto A = maker.dense_matrix(2, 2);
        auto x = maker.dense_vector(2);
        auto y = maker.dense_vector(2);
        auto alpha = maker.scalar<float>();
        auto beta = maker.scalar<float>();

        stream << A.dense_storage().copy_from(h_A.data());
        stream << x.dense_storage().copy_from(h_x.data());
        stream << alpha.dense_storage().copy_from(&h_alpha);
        stream << beta.dense_storage().copy_from(&h_beta);
        las->mv(y, alpha, A, x, beta);
        stream << y.dense_storage().copy_to(h_y.data());
        stream << synchronize();

        LUISA_INFO("{}: A = [1,2;3,4] x = [1,0]  y = A * x = [{},{}] ([1,3])", ID++, h_y[0], h_y[1]);

        las->mv(y, alpha, A.T(), x, beta);
        stream << y.dense_storage().copy_to(h_y.data());
        stream << synchronize();
        LUISA_INFO("{}: A = [1,2;3,4] x = [1,0]  y = A.T * x = [{},{}] ([1,2])", ID++, h_y[0], h_y[1]);
    }

    // batched mv
    {
        luisa::vector<float> h_A = {
            1.0f, 3.0f,// col 0
            2.0f, 4.0f // col 1
        };

        luisa::vector<float> h_x = {1.0f, 0.0f};
        auto h_y = luisa::vector<luisa::vector<float>>{2};
        std::for_each(h_y.begin(), h_y.end(), [](auto &item) { item.resize(2); });

        float h_alpha = 1.0f;
        float h_beta = 0.0f;

        auto A = maker.dense_matrix_batched(2, 2, 2);
        auto x = maker.dense_vector_batched(2, 2);
        auto y = maker.dense_vector_batched(2, 2);
        auto alpha = maker.scalar<float>();
        auto beta = maker.scalar<float>();

        // copy data to all batches
        for (auto &item : A.dense_storages()) stream << item.copy_from(h_A.data());
        for (auto &item : x.dense_storages()) stream << item.copy_from(h_x.data());
        stream << alpha.dense_storage().copy_from(&h_alpha);
        stream << beta.dense_storage().copy_from(&h_beta);
        las->mv_batched(y, alpha, A, x, beta);
        stream << synchronize();
        // copy data from all batches

        {
            int i = 0;
            std::for_each(h_y.begin(), h_y.end(),
                [&](auto &v) { stream << y.dense_storages()[i++].copy_to(v.data()); });
        }

        stream << synchronize();

        LUISA_INFO(
            "{}: A = [1,2;3,4] [1,2;3,4]  x = [1,0] [1,0]\n"
            "y = A * x = [{},{}] [{},{}] ([1,3] [1,3])",
            ID++, h_y[0][0], h_y[0][1], h_y[1][0], h_y[1][1]);
    }
}

void test_blas_level_3(tensor::TensorMaker &maker, tensor::LASInterface *las, Stream &stream) {
    LUISA_INFO("====================test blas level 3====================");
    auto ID = 0;
    {// mm C = alpha * A * B + beta * C
        luisa::vector<float> h_A = {
            1.0f, 3.0f,// col 0
            2.0f, 4.0f // col 1
        };

        luisa::vector<float> h_B = {
            1.0f, 3.0f,// col 0
            2.0f, 4.0f // col 1
        };

        luisa::vector<float> h_C = {
            0.0f, 0.0f,// col 0
            0.0f, 0.0f // col 1
        };

        float h_alpha = 1.0f;
        float h_beta = 0.0f;

        auto A = maker.dense_matrix(2, 2);
        auto B = maker.dense_matrix(2, 2);
        auto C = maker.dense_matrix(2, 2);
        auto alpha = maker.scalar<float>();
        auto beta = maker.scalar<float>();

        stream << A.dense_storage().copy_from(h_A.data());
        stream << B.dense_storage().copy_from(h_B.data());
        stream << C.dense_storage().copy_from(h_C.data());
        stream << alpha.dense_storage().copy_from(&h_alpha);
        stream << beta.dense_storage().copy_from(&h_beta);
        tensor::MatrixMulOptions options{
            .side = tensor::MatrixASide::LEFT};
        las->mm(C, alpha, A, B, beta, options);
        stream << C.dense_storage().copy_to(h_C.data());
        stream << synchronize();

        LUISA_INFO("{}: A = [1,2;3,4] B = [1,2;3,4]  C = A * B = [{},{}; {},{}] ([7,10; 15,22])", ID++, h_C[0], h_C[2], h_C[1], h_C[3]);

        las->mm(C, alpha, A.T(), B, beta, options);
        stream << C.dense_storage().copy_to(h_C.data());
        stream << synchronize();
        LUISA_INFO("{}: A = [1,2;3,4] B = [1,2;3,4]  C = A.T * B = [{},{}; {},{}] ([10,14; 14, 20])", ID++, h_C[0], h_C[2], h_C[1], h_C[3]);
    }

    // batched mm
    {
        luisa::vector<float> h_A = {
            1.0f, 3.0f,// col 0
            2.0f, 4.0f // col 1
        };

        luisa::vector<float> h_B = {
            1.0f, 3.0f,// col 0
            2.0f, 4.0f // col 1
        };

        luisa::vector<luisa::vector<float>> h_C = {
            // C0
            {
                0.0f, 0.0f,// col 0
                0.0f, 0.0f,// col 1

            },// C1
            {
                0.0f, 0.0f,// col 0
                0.0f, 0.0f,// col 1
            }};

        float h_alpha = 1.0f;
        float h_beta = 0.0f;

        auto A = maker.dense_matrix_batched(2, 2, 2);
        auto B = maker.dense_matrix_batched(2, 2, 2);
        auto C = maker.dense_matrix_batched(2, 2, 2);
        auto alpha = maker.scalar<float>();
        auto beta = maker.scalar<float>();

        // copy data to all batches
        for (auto &item : A.dense_storages()) stream << item.copy_from(h_A.data());
        for (auto &item : B.dense_storages()) stream << item.copy_from(h_B.data());

        stream << alpha.dense_storage().copy_from(&h_alpha);
        stream << beta.dense_storage().copy_from(&h_beta);
        tensor::MatrixMulOptions options{
            .side = tensor::MatrixASide::LEFT};
        las->mm_batched(C, alpha, A, B, beta, options);
        stream << synchronize();

        // copy data from all batches
        {
            int i = 0;
            std::for_each(h_C.begin(), h_C.end(), [&](luisa::vector<float> &v) { stream << C.dense_storages()[i++].copy_to(v.data()); });
        }
        stream << synchronize();

        LUISA_INFO("{}: A = [1,2;3,4] B = [1,2;3,4]  C = A * B = [{},{}; {},{}] [{},{}; {},{}] ([7,10; 15,22])",
                   ID++,
                   h_C[0][0], h_C[0][2], h_C[0][1], h_C[0][3],
                   h_C[1][0], h_C[1][2], h_C[1][1], h_C[1][3]);
    }
}

void test_sparse_level_1(tensor::TensorMaker &maker, tensor::LASInterface *las, Stream &stream) {
    LUISA_INFO("====================test sparse level 1====================");
    luisa::vector<float> h_values = {1.0f, 3.0f};
    luisa::vector<int> h_indices = {0, 2};
    auto x = maker.sparse_vector(3, h_values.size());

    luisa::vector<float> h_y = {1.0f, 2.0f, 3.0f};
    auto y = maker.dense_vector(h_y.size());

    float h_r;
    auto r = maker.scalar<float>();

    stream << x.sparse_vector_storage().values.copy_from(h_values.data());
    stream << x.sparse_vector_storage().indices.copy_from(h_indices.data());

    stream << y.dense_storage().copy_from(h_y.data());

    size_t buffer_size = las->spvv_buffer_size(r, y, x);
    auto ext_buffer = maker.external_buffer(buffer_size);
    las->spvv(r, y, x, ext_buffer);

    stream << r.dense_storage().copy_to(&h_r);
    stream << synchronize();

    LUISA_INFO("x = [1,0,3] y = [1,2,3] x dot y = {} (10)", h_r);
}

void test_sparse_level_2(tensor::TensorMaker &maker, tensor::LASInterface *las, Stream &stream) {
    LUISA_INFO("====================test sparse level 2====================");
    // A
    luisa::vector<float> h_values = {1.0f, 2.0f, 3.0f};
    luisa::vector<int> h_row_indices = {0, 1, 2};
    luisa::vector<int> h_col_indices = {0, 1, 2};
    auto A = maker.coo_matrix(3, 3, h_values.size());

    // x,y
    luisa::vector<float> h_x = {1.0f, 2.0f, 3.0f};
    luisa::vector<float> h_y = {0.0f, 0.0f, 0.0f};
    auto x = maker.dense_vector(h_x.size());
    auto y = maker.dense_vector(h_y.size());

    // alpha beta
    float h_alpha = 1.0f;
    float h_beta = 0.0f;
    auto alpha = maker.scalar<float>();
    auto beta = maker.scalar<float>();

    // copy A
    stream << A.sparse_matrix_storage().values.copy_from(h_values.data());
    stream << A.sparse_matrix_storage().i_data.copy_from(h_row_indices.data());
    stream << A.sparse_matrix_storage().j_data.copy_from(h_col_indices.data());

    // copy x y
    stream << x.dense_storage().copy_from(h_x.data());
    stream << y.dense_storage().copy_from(h_y.data());

    // copy alpha beta
    stream << alpha.dense_storage().copy_from(&h_alpha);
    stream << beta.dense_storage().copy_from(&h_beta);
    // mv
    size_t buffer_size = las->spmv_buffer_size(y, alpha, A, x, beta);
    auto ext_buffer = maker.external_buffer(buffer_size);
    las->spmv(y, alpha, A, x, beta, ext_buffer);
    stream << y.dense_storage().copy_to(h_y.data());
    stream << synchronize();

    LUISA_INFO("A = [1,0,0;0,2,0;0,0,3] x = [1,2,3]  y = A * x = [{}, {}, {}] ([1,4,9])", h_y[0], h_y[1], h_y[2]);
}

int main(int argc, char *argv[]) {
    Context context{argv[0]};

    // now only cuda
    Device device = context.create_device("cpu");
    Stream stream = device.create_stream();

    auto las = device.impl()->create_las_interface(stream.handle());

    tensor::TensorMaker maker{device, las};
    Buffer<uint64_t> buffer = device.create_buffer<uint64_t>(10);

    test_blas_level_1(maker, las, stream);
    test_blas_level_2(maker, las, stream);
    test_blas_level_3(maker, las, stream);
    test_sparse_level_1(maker, las, stream);
    test_sparse_level_2(maker, las, stream);

    device.impl()->destroy_las_interface(las);
}
