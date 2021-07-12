# gan(generative adversarial network)
gan, dcgan, wgan, cgan, celeba_cgan, unet, pix2pix, cyclegan

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

### dataset
mnist(28x28 images), 10 classes

### component
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
 G   D  
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
use dcgan

### fundamentals
convolution  
transposed convolution

### concept
generate images of much improved quality  
a strong candidate for unsupervised learning  
walk in the latent space

### dataset
mnist(28x28 images), 10 classes

### component
#### generator
generator block(transposed convolution, batch norm, relu, tanh)

#### discriminator 
discriminator block(convolution, batch norm, leakyrelu)

#### loss
BCE(binary cross entropy)  
generator loss  
discriminator loss

## wgan(wasserstein gan)
### problem of vanilla gan
unstable training  
use wgan with gradient penalty

### fundamentals
entropy, information theory, cross entropy

### concept
improves the stability of gan training  
resolve mode collapsing  
avoid gradient vanishing

#### problem1
gradient vanishing  
because of cross entropy distribution distance

#### solution1
wasserstein distance(earth mover's distance)  
discriminator -> critic

loss function  
min max E[C(x)] - E[C(G(z))]  
 G   C  
real data distribution : E[C(x)]  
fake data distribution : E[C(G(z))]

#### problem2
gradient exploding

#### solution2
gradient penalty

lipschitz function : |dy/dx| <= K  
1-lipschitz continuity : (k = 1) |dy/dx| <= 1  
weight clipping  
regularization  
interpolation G(z), x -> creation x^ -> G(z) quality improvement -> critic train speed control

### dataset
mnist(28x28 images), 10 classes

### component
#### generator
generator block(transposed convolution, batch norm, relu, tanh)

#### critic
critic block(convolution, batch norm, leakyrelu)

#### gradient penalty

#### loss
generator loss  
critic loss

## cgan(conditional gan)
control the results of gan by assigning conditions

### problem of vanilla gan
uncontrollable result  
use cgan

### concept
input : latent vector(z)(= noise vector) + condition vector(y)(= one-hot vector)  
discriminator : D(x|y)  
generator : G(z|y)

### dataset
mnist(28x28 images), 10 classes

### component
#### generator
generator block(transposed convolution, batch norm, relu, tanh)

#### discriminator
discriminator block(convolution, batch norm, leakyrelu)

#### input
latent vector, one-hot vector concatenation

#### loss
min max V(D, G)  
 G   D  
D(x|y), G(z|y)

## celeba_cgan
21st label name : male  
1 -> male  
0 -> female  
21st label one-hot vector

### dataset
celeba(64x64x3 images), 40 labels

## pix2pix
photorealistic, stylistic

### concept
#### example
labels to street scene  
labels to facade  
black & white to color  
aerial to map  
day to night  
edges to photo

### structure
#### generator(G)
input(cgan generator - noise vector) -> pix2pix generator -> generated output  
x -> G(x)

#### discriminator(D)
input(real input + (real output or generated output)) -> pix2pix discriminator -> real/fake  
x, G(x) -> fake  
x, y -> real

### dataset
VOCSegmentation

### component
generator
u-net
encoder-decoder
deconv-net
convolution, transposed convolution
rgb image(input)(x) -> encoder -> latent vector -> decoder -> segmentation mask(output)(y)

skip connection
forward pass : encoder's information -> decoder
backward pass : encoder's gradient flow improvement

encoder block(convolution, batch norm, leakyrelu)
x(256 x 256 x 3) -> E(x)(1 x 1 x 512)

decoder block(transposed convolution, batch norm, relu)
E(x)(1 x 1 x 512) -> y(256 x 256 x 3)
dropout

discriminator
patchgan

loss
adversarial loss
pixel distance loss

paired image

unpaired image-to-image translation

## unet


## cyclegan


## gaugan
nvidia ai image creator
