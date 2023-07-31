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

        stream << x.copy_from(h_x.data());
        las->nrm2(r, x);
        stream << r.copy_to(&h_r);
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

        stream << x.copy_from(h_x.data());
        stream << y.copy_from(h_y.data());
        las->dot(r, x, y);
        stream << r.copy_to(&h_r);
        stream << synchronize();
        LUISA_INFO("{}: dot([1,2,3],[4,5,6])={} (32)", ID++, h_r);
    }

    {// Iamax & Iamin

        // host
        luisa::vector<float> h_x = {2.0f, 4.0f, 6.0f};
        int h_r;

        // device
        auto r = maker.scalar<int>();
        auto x = maker.dense_vector(h_x.size());

        stream << x.copy_from(h_x.data());
        las->Iamax(r, x);
        stream << r.copy_to(&h_r);
        stream << synchronize();
        LUISA_INFO("{}: Iamax([2,4,6])={} (3)", ID++, h_r);

        las->Iamin(r, x);
        stream << r.copy_to(&h_r);
        stream << synchronize();
        LUISA_INFO("{}: Iamin([2,4,6])={} (1)", ID++, h_r);
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

        stream << A.copy_from(h_A.data());
        stream << x.copy_from(h_x.data());
        stream << alpha.copy_from(&h_alpha);
        stream << beta.copy_from(&h_beta);
        las->mv(y, alpha, A, x, beta);
        stream << y.copy_to(h_y.data());
        stream << synchronize();

        LUISA_INFO("{}: A = [1,2;3,4] x = [1,0]  y = A * x = [{},{}] ([1,3])", ID++, h_y[0], h_y[1]);

        las->mv(y, alpha, A.T(), x, beta);
        stream << y.copy_to(h_y.data());
        stream << synchronize();
        LUISA_INFO("{}: A = [1,2;3,4] x = [1,0]  y = A.T * x = [{},{}] ([1,2])", ID++, h_y[0], h_y[1]);
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

        stream << A.copy_from(h_A.data());
        stream << B.copy_from(h_B.data());
        stream << C.copy_from(h_C.data());
        stream << alpha.copy_from(&h_alpha);
        stream << beta.copy_from(&h_beta);
        tensor::MatrixMulOptions options{
            .side = tensor::MatrixASide::LEFT
        };
        las->mm(C, alpha, A, B, beta, options);
        stream << C.copy_to(h_C.data());
        stream << synchronize();

        LUISA_INFO("{}: A = [1,2;3,4] B = [1,2;3,4]  C = A * B = [{},{}; {},{}] ([7,10; 15,22])", ID++, h_C[0], h_C[2], h_C[1], h_C[3]);

        las->mm(C, alpha, A.T(), B, beta, options);
        stream << C.copy_to(h_C.data());
        stream << synchronize();
        LUISA_INFO("{}: A = [1,2;3,4] B = [1,2;3,4]  C = A.T * B = [{},{}; {},{}] ([10,14; 14, 20])", ID++, h_C[0], h_C[2], h_C[1], h_C[3]);
    }
}

int main(int argc, char *argv[]) {
    Context context{argv[0]};

    // now only cuda
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    auto las = device.impl()->create_las_interface(stream.handle());

    tensor::TensorMaker maker{device};

    test_blas_level_1(maker, las, stream);
    test_blas_level_2(maker, las, stream);
    test_blas_level_3(maker, las, stream);

    device.impl()->destroy_las_interface(las);
}
