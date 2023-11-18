using Flux
using MLDatasets
using Flux.Losses: mse
using Flux.Optimise: ADAM
using Base.Iterators: partition
using Images, ImageMagick
using Random


# 데이터셋 로드
Random.seed!(123)
data = MNIST.traindata(Float32)

# U-Net 모델 정의

function SimpleUnet()
    image_channels = 1
    down_channels = (64, 128, 256, 512, 1024)
    up_channels = (512, 256, 128, 64)
    out_dim = 1

    # 다운샘플링 블록
    down_blocks = []
    for (i, out_channels) in enumerate(down_channels)
        if i == 1
            push!(down_blocks, Chain(Conv((3, 3), image_channels => out_channels, pad=(1, 1), relu), Conv((3, 3), out_channels => out_channels, pad=(1, 1), relu)))
        else
            push!(down_blocks, Chain(Conv((3, 3), down_channels[i-1] => out_channels, pad=(1, 1), relu), Conv((3, 3), out_channels => out_channels, pad=(1, 1), relu)))
        end
        push!(down_blocks, MaxPool((2, 2)))
    end

    # 업샘플링 블록
    up_blocks = []
    for (i, out_channels) in enumerate(up_channels)
        if i == 1
            push!(up_blocks, Chain(ConvTranspose((2, 2), down_channels[end] => out_channels, stride=(2, 2)), Conv((3, 3), out_channels => out_channels, pad=(1, 1), relu), Conv((3, 3), out_channels => out_channels, pad=(1, 1), relu)))
        else
            push!(up_blocks, Chain(ConvTranspose((2, 2), up_channels[i-1] => out_channels, stride=(2, 2)), Conv((3, 3), out_channels => out_channels, pad=(1, 1), relu), Conv((3, 3), out_channels => out_channels, pad=(1, 1), relu)))
        end
    end

    # 출력 레이어
    output = Conv((1, 1), up_channels[end] => out_dim)

    return Chain(down_blocks..., up_blocks..., output)
end

# 블록 정의
struct Block
    conv1::Conv
    conv2::Conv
    bnorm1::BatchNorm
    bnorm2::BatchNorm
    relu::Function
end

function Block(in_channels, out_channels, time_emb_dim, up=false)
    conv1 = Conv((3, 3), in_channels=in_channels, out_channels=out_channels, pad=(1, 1))
    conv2 = Conv((3, 3), in_channels=out_channels, out_channels=out_channels, pad=(1, 1))
    bnorm1 = BatchNorm(out_channels)
    bnorm2 = BatchNorm(out_channels)
    relu = x -> max.(x, 0)
    return Block(conv1, conv2, bnorm1, bnorm2, relu)
end

function (block::Block)(x, t)
    # 첫 번째 컨볼루션
    h = block.relu(block.bnorm1(block.conv1(x)))

    # 시간 임베딩
    time_emb = block.relu(block.time_mlp(t))
    time_emb = reshape(time_emb, (1, size(time_emb, 1), 1, 1))
    time_emb = repeat(time_emb, (size(h, 1), 1, size(h, 3), size(h, 4)))
    h = h + time_emb

    # 두 번째 컨볼루션
    h = block.relu(block.bnorm2(block.conv2(h)))

    # 다운샘플링 또는 업샘플링
    return block.transform(h)
end

function get_loss(model, x_0, t)
    x_noisy, noise, t_float = forward_diffusion_sample(x_0, t)
        noise_pred = model(x_noisy, t)
        return Flux.mse(noise, noise_pred)
    end

function forward_diffusion_sample(image, t)
    noise = randn(size(image))
    t_float = Float64.(t)  # 시간 정보를 실수 형식의 배열로 변환
    t_float = reshape(t_float, 1, 1, length(t_float)) # t_float를 3차원 배열로 변환
    noisy_image = image .+ sqrt.(t_float) .* noise
 # 브로드캐스팅을 사용하여 잡음을 이미지에 추가
    return noisy_image, noise, t_float
end
# 잡음 생성 함수 정의



# 학습 함수 정의

function train_diffusion_model(model, data, iterations, batch_size, display_interval)
    timesteps = 100
    betas = range(0.0001, 0.02, length=timesteps)

    # 데이터 배치 생성
    batches = partition(data, batch_size)
    opt = ADAM(0.001)

    iteration = 0
    for batch in batches
        Flux.reset!(opt)  # 옵티마이저의 그래디언트 초기화

        t = rand(1:timesteps, size(batch, 4))
        loss = get_loss(model, batch, t)
        Flux.back!(loss)
        Flux.update!(opt, params(model))  # 옵티마이저를 사용하여 모델 파라미터 업데이트

        # 결과 출력
        iteration += 1
        if iteration % display_interval == 0
            display_results(model, data)  # 결과 출력 함수 호출
        end

        # 지정된 반복 횟수(iterations)에 도달하면 종료
        if iteration >= iterations
            break
        end
    end
end

function display_results(model, data)
    # 결과 이미지 출력 코드 작성
    # 예시: 테스트 데이터에서 이미지를 선택하여 모델에 입력하고 결과 이미지를 출력하는 코드
    test_data = MNIST.testdata(Float32)
    image_index = rand(1:length(test_data[1]))
    input_image = test_data[1][:, :, image_index]
    target_image = test_data[2][image_index]

    # 입력 이미지를 모델에 전달하여 결과 이미지 생성
    result_image = model(input_image)

    # 입력 이미지, 결과 이미지, 정답 이미지를 함께 시각화하여 출력
    plot(
        heatmap(input_image, color=:grays, title="Input Image"),
        heatmap(result_image, color=:grays, title="Result Image"),
        heatmap(target_image, color=:grays, title="Target Image"),
        layout=(1, 3)
    )
end


# 훈련
model = SimpleUnet()
train_diffusion_model(model, data, 1000, 128, 100)
