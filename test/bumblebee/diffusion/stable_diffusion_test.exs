defmodule Bumblebee.Diffusion.StableDiffusionTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  describe "integration" do
    test "text_to_image/6" do
      repository_id = "CompVis/stable-diffusion-v1-4"

      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})

      {:ok, clip} = Bumblebee.load_model({:hf, repository_id, revision: "fp16", subdir: "text_encoder"})

      {:ok, unet} =
        Bumblebee.load_model({:hf, repository_id, revision: "fp16", subdir: "unet"},
          params_filename: "diffusion_pytorch_model.bin"
        )

      {:ok, vae} =
        Bumblebee.load_model({:hf, repository_id, revision: "fp16", subdir: "vae"},
          architecture: :decoder,
          params_filename: "diffusion_pytorch_model.bin"
        )

      {:ok, scheduler} = Bumblebee.load_scheduler({:hf, repository_id, revision: "fp16", subdir: "scheduler"})

      {:ok, featurizer} =
        Bumblebee.load_featurizer({:hf, repository_id, revision: "fp16", subdir: "feature_extractor"})

      {:ok, safety_checker} = Bumblebee.load_model({:hf, repository_id, revision: "fp16", subdir: "safety_checker"})

policy = Axon.MixedPrecision.create_policy(compute: :f16, params: :f16, output: :f16)
clip = %{clip | model: Axon.MixedPrecision.apply_policy(clip.model, policy)}
unet = %{unet | model: Axon.MixedPrecision.apply_policy(unet.model, policy)}
vae = %{vae | model: Axon.MixedPrecision.apply_policy(vae.model, policy)}

safety_checker = %{
  safety_checker
  | model: Axon.MixedPrecision.apply_policy(safety_checker.model, policy)
}


      serving =
        Bumblebee.Diffusion.StableDiffusion.text_to_image(clip, unet, vae, tokenizer, scheduler,
          num_steps: 2,
          safety_checker: safety_checker,
          safety_checker_featurizer: featurizer,
          defn_options: [compiler: EXLA]
        )

      prompt = "numbat in forest, detailed, digital art"

      assert %{
               results: [%{image: %Nx.Tensor{}, is_safe: _boolean}]
             } = Nx.Serving.run(serving, prompt)
    end
  end
end
