# gan(generative adversarial network)
gan, dcgan, wgan, cgan, celeba cgan, unet, pix2pix, cyclegan

## requirements
google colab  
pytorch  
torchvision

## vanilla gan
### fundamentals
나의 행운은 적의 불행이요, 나의 불행은 적의 행운이다.

#### background
adversarial search  
ex) mini-max search for tic-tac-toe

discriminative model(classifier)  
svm -> cnn

generative model  
latent vector -> complex model, realistic image  
AE(auto encoder) -> VAE(variational auto encoder)

### concept
나는 강해진다, 나의 적이 강한 만큼

#### generator
G : p_z(z) -> p_g  
trying to produce fake currency

#### discriminator
D : D(x;θ_d) -> fake/real  
x ~ p_g -> fake, x ~ data -> real  
trying to detect the counterfeit currency

#### train
opposition, competition of generator, discriminator

#### test
trained, saved parameter θ_g  
saved generator G(θ_g)  
input : latent vector z_k  
output : various sample creation G(z_k)

### component
#### dataset
mnist(28x28 image)  
10 classes

#### generator(G = θ_g)
decoder  
latent vector(z, p_z) -> synthesized image(G(z), G(z;θ_g), p_g)  
unused convolution in vanila gan  
generator block(linear, batch norm, relu), fully connected, sigmoid  
input dimension : 10  
output dimension : 784

#### discriminator(D = θ_d)
encoder, classifier  
real data(x), fake data(G(z)) -> real/fake  
discriminator block(linear, relu), fully connected  
input dimension : 784  
output dimension : 1

#### loss
BCE(binary cross entropy)  
generator loss  
discriminator loss

#### optimizer
adam

### train purpose
P_data (= x) = P_z (= G(z))

### train difficulty
min max V(G, D)  
gradient descent application difficulty  
heuristic train strategy

### train process
#### before training
G is not trained -> G(z) & x do not match  
D is not trained -> D is unstable

#### D is trained
G is not trained -> G(z) & x do not match  
D is trained -> D is stable

#### G is training
G is training -> G(z) approaches x  
D is trained -> D is stable

#### G & D are trained
G is trained -> G(z) & x match  
D is trained -> D is uniform (1/2)

## dcgan(deep convolutional gan)
### problem of vanilla gan
generates images, but not visually pleasing results

### fundamentals
convolution  
transposed convolution

### conept
generate images of much improved quality  
a strong candidate for unsupervised learning  
walk in the latent space





wgan

unstable training
use wgan with gradient penalty

cgan

uncoltrollable result
use cgan

celeba cgan
unet
pix2pix
cyclegan



celeba


