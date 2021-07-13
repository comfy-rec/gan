# gan(generative adversarial network)
gan, dcgan, wgan, cgan, celeba_cgan, pix2pix, unet, cyclegan

## requirements
google colab  
pytorch  
torchvision

## vanilla gan
### fundamentals
my fortune is the enemy's misfortune, my misfortune is the enemy's fortune.  
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
I become strong, as strong as my enemy is.  
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
min max V(G, D)  
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
photorealistic, stylistic, paired image

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
#### generator
u-net  
encoder-decoder  
deconv-net  
convolution, transposed convolution  
rgb image(input)(x) -> encoder -> latent vector -> decoder -> segmentation mask(output)(y)

skip connection  
forward pass : encoder's information -> decoder  
backward pass : encoder's gradient flow improvement

encoder block(convolution, batch norm, leakyrelu)  
x(256x256x3) -> E(x)(1x1x512)

decoder block(transposed convolution, batch norm, relu)  
E(x)(1x1x512) -> y(512x512x3)

#### discriminator
patchgan

#### loss
pix2pix loss = cgan loss(adversarial loss, gan loss) + pixel distance loss  
min max L_cgan(G, D) + λL_L1(G)  
 G   D  
L_cgan(G, D) : adversarial loss  
λL_L1(G) : pixel distance loss  
y : groundtruth

### realization
#### generator -> unet
feature map, contract, expand, sigmoid
#### contractor
contracting block(convolution, leakyrelu, maxpool, batch norm, dropout)
#### crop
#### expander
expanding block(upsample, convolution, relu, batch norm, dropout)
#### feature map
feature map block(convolution)

#### discriminator
feature map, contract

#### loss
BCEWithLogitsLoss, L1Loss

## unet
### component
#### contractor
contracting block(convolution, relu, maxpool)
#### crop
#### expander
expanding block(upsample, convolution, relu)

## cyclegan
unpaired image-to-image translation, unpaired image domain, style gan
monet <-> photo
zebra <-> horse
summer <-> winter

### background
paired image domains -> ground truth o
unpaired image domains -> ground truth x

### concept
2 domains : 2 gan : 2 cycle consistency
2 gan
gan_(Z->H), gan_(H->Z)
same structure, different training

### structure
2 gan
gan_(Z->H), gan_(H->Z)

gan_(Z->H) = generator : G_(Z->H), discriminator : (D_H)
gan_(H->Z) = generator : G_(H->Z), discriminator : (D_Z)

generator(G_(Z->H), G_(H->Z)) = same structure, different training
discriminator((D_H), (D_Z)) = same structure, different training

### dataset
horse2zebra(3x256x256 image data)

### component
#### generator
unet + dcgan + skip-connection
featuremap block(convolution)
contractor block(convolution, relu, leakyrelu, instancenorm)
residual block(convolution, instancenorm, relu)
expander block(transposed convolution, instancenorm, relu)
tanh

#### discriminator
patchgan(3x256x256 -> 1x8x8)
featuremap block(convolution)
contractor block(convolution, leakyrelu)
convolution

#### loss
total loss = adversarial loss + cycle consistency loss + identity loss
total loss : L(G_(Z->H), G_(H->Z), D_H, D_Z)
adversarial loss : L_gan(G_(Z->H), D_H, Z, H) + L_gan(G_(H->Z), D_Z, Z, H)
cycle consistency loss : λ_1L_cyc(G_(H->Z), G_(Z->H))
identity loss : λ_2L_id(G_(H->Z), G_(Z->H))

#### adversarial loss
improve the quality
discriminator loss
generator loss
MSELoss, L1Loss

gan_(Z->H) loss
generator : G_(Z->H), discriminator : (D_H)
L_gan(G_(Z->H), D_H, Z, H)

gan_(H->Z) loss
generator : G_(H->Z), discriminator : (D_Z)
L_gan(G_(H->Z), D_Z, Z, H)

#### cycle consistency loss
avoide mode collapse
H->Z->H loss + Z->H->Z loss

#### identity loss
keep color
pixel distance
G_(Z->H), horse
G_(H->Z), zebra

## gaugan
nvidia ai image creator
