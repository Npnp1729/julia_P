using ImageIO, Images, ImageFiltering

function diffusion_model(input_image_path, output_image_path)
    # 이미지 로딩
    input_image = load(input_image_path)

    # 이미지를 흐리게 만들기 위한 필터 적용
    blurred_image = imfilter(input_image, Kernel.gaussian(10))

    # 이미지 저장
    save(output_image_path, blurred_image)
end

# 입력 이미지 경로와 출력 이미지 경로 설정

input_image_path = "example.jpg"
output_image_path = "output_image.jpg"

# Diffusion 모델 실행
diffusion_model(input_image_path, output_image_path)