## Abstract
Variational Auto Encoder는 X에 대한 latent vector를 찾아내기 위해 latent vector를 다루기 쉬운 확률분포라고 가정을 하고 모델을 훈련시킨다. VAE의 이러한 가정은 실제 X에 대한 분포가 사전에 가정한 확률분포와 차이가 있다는 점에서 데이터의 재현율이 떨어진다는 단점이 있다. VAE의 이러한 단점을 극복하고 성능을 개선시키기 위해  VAE를 기반으로한 여러 최신 모델 중 IAF-VAE, IWAE, AAE 모델을 살펴본다. large-scale 이미지 데이터셋인 COCO에 적용하여 각각의 방법들이 실험 결과에서 어떠한 차이가 있는지에 대해 비교 분석하여 성능을 평가한다. 

## Introduction
Variational Auto Encoder[1] 는 X에 대한 latent vector를 찾아내기 위해 X에 대한 확률분포를 다루기 쉬운 확률분포로 가정을 하고 생성 모델을 훈련시킨다. 일반적으로  VAE의 latent vector는 정규 분포를 따른다고 가정을 한다. 정규분포는 자연현상에서 관찰되는 일반적인 분포이며, 평균과 분산인 2개의 파라미터로 분포를 명시적으로 나타낼 수 있다는 장점이 있다. 하지만 X에 대한 확률 분포에 대한 가정은 관찰 가능한 X의 분포와 차이가 있다. 따라서 VAE 모델은 낮은 데이터 재현율을 보이며 이로 인해 VAE 기반 모델로부터 생성된 데이터는 흐리게 생성된다는 한계점을 보인다.
이러한 VAE의 단점을 아래 두가지 방법으로 개선시키고자 한다. 
latent vector의 분포를 실제 데이터와 더욱 가깝게 근사시키는 방법
Loss를 계산하는 전혀 새로운 기법을 적용하는 방법 
첫 번째 방법으로 우리는 Normalization Flow를 적용한 IAF-VAE[2]와 가중치를 중요도에 따라 달리 적용한 IWAE를 살펴보고자 한다. IAF-VAE는 Inverse Autoregressive Flows VAE의 약자로 latent vector의 분포를 현실 데이터에 맞추기 위해 단계적으로 분포를 변화 시키는 Normalization Flow를 적용한 모델이다. IAF-VAE는 VAE가 가지고 있는 근본적인 문제인 “latent vector의 확률 분포를 가정한다”를 해결하는 모델이다.
두 번째 방법으로 우리는 GAN의 학습 방법인 적대적 생성 기법(Adversarial Generation)을 VAE에 적용한다. AAE[3]는 Adversal Auto Encoder의 약자로 VAE에 GAN에서 모델을 학습하는 방법인 적대적 학습 방법을 적용하여 Loss를 계산한다. AAE는 discriminator를 이용하여 latent vector의 분포가 사전 확률 분포와 최대한 일치하도록 모델 성능을 개선한 모델이다.
본 보고서에선 VAE를 베이스라인으로 두고, IAF-VAE, IWAE 그리고 AAE를 동일한 조건하에서 비교 분석하여 IAF-VAE, IWAE 그리고 AAE가 VAE의 단점을 어떻게 극복했는지 COCO 데이터셋을 이용하여 실험한다. 모델 간의 성능은 MSE를 기준으로 평가하며, 생성된 이미지 샘플을 출력하여 정량 및 정성적 평가를 한다.  

## Methods

IAF-VAE: IAF-VAE는 latent vector z를 normalization flow를 이용해 변형 시키는 모델

IWAE: VAE의 ELBO를 실제 log likelihood에 조금 더 가깝게 만들기 위해 더 많은 sample을 loss function으로 이용한 모델

AAE: VAE가 regularization을 qΦ(z|x)와 p(z)의 KL divergence를 사용하여 최적화 하는 반면에, AAE는 p(z)에서 만든 진짜 샘플과 q(z)에서 만든 가짜 샘플을 유사하게 만들어 학습.


3.1 VAE
본 실험에서 VAE 모델은 베이스라인으로서의 역할을 하며, Encoder는 단순한 Convolution 모델을 사용했고, Decoder는 단순한 Convolution Transpose 모델을 사용했다. 
3.2 IAF-VAE
IAF-VAE는 VAE의 latent vector에 normalization flow를 적용한 모델이다. 일반적으로 Gaussian이라고 가정하는 z의 분포를 우리는 알 수 없는 데이터의 분포로 변환시킬 수 있어 보다 나은 성능을 보일 것으로 기대한다. IAF-VAE를 사용할 경우 일반적인 VAE보다 많은 메모리를 사용하기 때문에 상대비교를 하였다.
3.3 IWAE
Baseline 모델인 VAE를 사용하되, Importance weight를 적용하기 위해 Loss function을 다르게 주었다. 이를 위해서 latent vector인 z를 원하는 sample 수 만큼 늘린뒤, sample의 가중치를 사용해 Lower bound를 계산하게 된다. 이때, sample의 수 만큼 NN을 늘리지 않고 latent vector z를 sample의 수 만큼 복제를 하게 된다. 논문의 저자들은 기대치를 이용하는 것만으로도 sample 가중치를 계산하는데 무리가 없다는 것을 보여주었기 때문에, 연산 효율성 측면에서 복제라는 테크닉을 사용했다.
3.4 AAE
VAE의 모델을 그대로 사용하는데, encoder가 Generator 역할을 하고, 별도의 Discriminator 를 붙여 학습한다. Prior(Target Distribution)에서 추출한 샘플과 Generator에서 생성한 샘플이 최대한 가깝게 만드는데, VAE에서 KL divergence를 계산하는 대신에 Discriminator가 해당 역할을 한다. Discriminator는 실제 이미지를 1로 반환하고 Generator에서 생성한 이미지를 0으로 반환하여 output 단계에서 sigmoid function을 두어 Binary classification 한다. Generator는 인코더에서 생성한 mu와 sigma로 z를 샘플링하여 Discriminator가 1을 반환하도록 학습한다. 

## Experimental setting
4.1 Dataset
VAE와 VAE 기반으로 제안된 최신 모델간의 성능 평가를 위해 large-scale 이미지 데이터 셋 COCO dataset[4]을 실험에 사용한다. COCO 데이터 셋은 80개여개 이상의 카테고리에 해당하는 150만개의 데이터로 구성된다. 그 중 선별한 약 10,000개의 data를 train set으로 사용하고, 2,000개의 data를 test set으로 사용한다.

4.4 Evaluation
이번 실험에는 원 데이터의 재현율에 대한 성능을 중점적으로 4개의 각 모델 성능을 비교 평가 하고자 한다. 따라서 원 데이터간와 생성 데이터 간의 오차를 나타내는 Mean Square Error 식을 사용한다. MSE식의 정의는 아래와 같다. 재현율 뿐만 아니라 모델의 계산 소요 시간을 함께 고려해 large-scale의 이미지시 4개의 모델이 얼마나 효율적으로 이미지의 특징을 배우고 생성해 낼수 있지를 종합적으로 고려한다. 

VAE 모델은 아무것도 배우지 않은 상태이기 때문에 흐린 이미지를 생성한다. Epoch= 5일시, 전체적인 윤곽을 제대로 생성하고, 색깔도 어느정도 재현하지만 배경에 해당되는 세세한 표현은 하지 못한다.  Epoch= 50일시, 전보다 뚜렷한 윤곽과 색채를 재현하는 것을 볼 수 있지만 여전히 흐린 이미지를 생성하는 한계점을 보인다.

IWAE는 VAE보다 조금 더 선명한 이미지를 생성해냈다. 그 이유는 IWAE가 VAE보다 log-likelihood에 조금 더 가까운 lower bound를 갖기 때문이다. 만약 IWAE의 sample 수가 많으면 많을 수록 더욱 더 우수한 성능을 보일 것이다. 하지만 연산 성능의 한계로 많은 수의 sample을 사용할 수 없으며, 연산 속도를 기준으로 IWAE는 가장 안좋은 결과를 보였기 때문에, 생성 모델 측면에서 뚜렷한 한계를 보인다.

AAE는 동일한 epoch 수준에서 VAE와 비교 해보았을때, 초기부터 훨씬 더 나은 샘플 이미지를 생성해낸다. AAE 모델이 epoch=5인 생성 이미지와 VAE 모델이 epoch=100 생성 이미지는 정성적으로 품질이 동일하거나 AAE가 형체 구현 면에서 훨씬 더 나은 것을 볼 수 있다.  앞전의 IAF-VAE에 비해 색체도 거의 원본 이미지와 유사한 수준으로 구현하였다. 다만 최신 GAN 을 기반으로한 고화질 이미지 생성 모델에 비해 AAE는 모든 이미지를 완벽하게 재현하는 것에 한계점이 있지만, 현재 비교 대상인 VAE 기반 모델 중에서 가장 우수한 성능을 보인다.

IAF-VAE는 최종적으로 VAE에 비해 좀더 나은 MSE를 보이는 것을 확인할 수 있다. 이는 z의 latent vector를 Gaussian이 아닌 현실 데이터에 맞게 변환 시키는데서 기인한다. 하지만 AAE나 IWAE와 같은 모델에 비해 최종 MSE는 떨어지지만 생성 이미지 내에서 물체의 형체를 잘 구현해낸다는 점에 MSE로만 성능을 평가하기엔 한계점이 있다. 다만, VAE에 비해 학습을 위한 cost가 많이 소요되는 것으로 파악되는데, normalization flow를 하는 과정에서 MADE 아키텍처가 전체 모델에서 상당한 비중을 차지한다.

## Conclusion
본 프로젝트에서는 large-scale 이미지인 COCO 데이터셋을 이용하여 VAE 기반의 다양한 개선 모델을 적용하여 성능을 비교 평가해보았다. 실험의 대상이 되는 모델은 VAE, IWAE, IAF-VAE, AAE으로, 비교 평가를 위해 동일한 hyper-parameter를 설정해 실험하였다. 이 실험에서 모델의 평가로서 1) MSE를 기반한 데이터 재현율 평가, 2) 생성 이미지 샘플에 대한 정성적 평가, 3) 계산 비용에 대한 평가를 진행하였다. MSE 기반 평가로서 AAE > IWAE> IAF-VAE > VAE 모델 순으로 성능을 보였다. 생성 이미지에 대한 정성적 평가 부분에서 AAE의 샘플 이미지는 품질이 우수하였다. 또한, IAF-VAE는 다른 모델에 비해 높은 MSE임에도 불구하고 물체의 형체를 선명하게 구현했으나, 색과 톤 부분에서 원본 이미지와 차이를 보였다. 계산 소요 시간은 AAE > VAE >  IAF-VAE > IWAE 순으로 으로 AAE는 VAE보다 계산 소요 시간이 줄어들었지만 IAF-VAE, IWAE 모델은 성능 개선 만큼 소요시간도 VAE보다 좀더 요구 되었다. 성능평가에 대한 결론으로서 데이터 재현율, 생성이미지품질, 계산효율성 부분에서 AAE가 베이스라인 모델인 VAE에 비해 월등한 성능 개선을 보였다. 


### References

[1] Auto-Encoding Variational Bayes, D. Kingma and M. Welling, ICLR 2014, https://arxiv.org/pdf/1312.6114.pdf
[2] Improving Variational Inference with Inverse Autoregressive Flow, D. Kingma et al, NeurIPS 2016, https://arxiv.org/pdf/1606.04934.pdf
[3] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "Importance weighted autoencoders." arXiv preprint arXiv:1509.00519 (2015).
[4] AVAE: Adversarial Variational Auto Encoder, https://arxiv.org/abs/2012.11551
[5] COCO Dataset, https://cocodataset.org/
