# Style Transfer using Convolutional Neural Network
###### Ryan Chan Last Updated: 17 October 2018


## Motivation
#### Research Motivation
Layers in neural network contains useful information. For example, one can use convolutional operation to reduce the dimension of the data, while embedding common information between each layer. Formerly known actviation maps, they contain useful presentations that can be processed for further purpose. Artistic Style Transfer is one of many examples that utilizes actvations in convolutional neural networks. This project sets to explore activation maps further. 

#### Personal Motivation
The artistic and imaginative side of human is known to be one of the most challenging perspective of life to model. Due to its free form and huamnly-cultivated experience, art is often appreciated not only because of its visual apperance, but also the history and motivations of the artist. In this project, I attempt to answer this question: "If we were to create a model that creates art, how would it do it, and what separates that from human life?" 

This is my first project look in-depth into an academic paper and attempt to implement the model from scratch. Because it was widely used to illustrate what neural networks can do, artistic style transfer remains as one of the most interesting beginner projects. I am doing this to cultivate my extensive and critical thinking sills, and also understand the model thoroughly, to the extent where I have no doubt if asked to explain how it works from zero to a hundred. 

## Model Structure and the Flow of Information
_Given a content image and a style image, how does the style transfer model generate a synthsized image?_

First, the style image is rescaled to be the same size as the content image. Both are passed into a pre-trained VGG Network, the activations of specific layers are extracted as constants. For style image, we pre-computed the gram matrix for each style activation. Then, we initialize a tensorflow variable with a random image that has the same size as the content image. We also build the tensorflow graph that defines the strucutre for how content loss and style loss is computed. Then, the image is updated iteratively by computing the gradient of scaled content and style losss combined. 

_How are style loss and content loss computed?_

The paper defines the style loss to be $$L_{\text{style}} = $$

and content loss to be $$L_{\text{content}}$$


## Replication of Figures
### Figure 1 - Image Representations in a Convolutional Neural Network

**Content Reconstruction.**
The following figures are created with `alpha = 1, beta = 0`.

|<img src="images/figures/fig1/cont1.jpg" alt="fig1_cont1">|<img src="images/figures/fig1/cont2.jpg" alt="fig1_cont2">|<img src="images/figures/fig1/cont3.jpg" alt="fig1_cont3">|<img src="images/figures/fig1/cont4.jpg" alt="fig1_cont4">|<img src="images/figures/fig1/cont5.jpg" alt="fig1_cont5">|
|:---:|:---:|:---:|:---:|:---:|
|`conv1_1`|`conv2_1`|`conv3_1`|`conv4_1`|`conv5_1`|

**Style Reconstruction.**
The following figures are created with `alpha = 0, beta = 1`.

|<img src="images/figures/fig1/styl1.jpg" alt="fig1_styl1">|<img src="images/figures/fig1/styl2.jpg" alt="fig1_styl2">|<img src="images/figures/fig1/styl3.jpg" alt="fig1_styl3">|<img src="images/figures/fig1/styl4.jpg" alt="fig1_styl4">|<img src="images/figures/fig1/styl5.jpg" alt="fig1_styl5">|
|:---:|:---:|:---:|:---:|:---:|
|`conv1_1`|`conv1_1`<br>`conv2_1`|`conv1_1`<br>`conv2_1`<br>`conv3_1`|`conv1_1`<br>`conv2_1`<br>`conv3_1`<br>`conv4_1`|`conv1_1`<br>`conv2_1`<br>`conv3_1`<br>`conv4_1`<br>`conv5_1`|

### Figure 3 - Well-known Artwork examples
The following figures are created with: <br>
Weights:  `alpha = 1e-6, beta = 1` <br>
Content Layers: `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1`<br>
Style Layers: `conv4_2`<br>

|<img src="images/figures/fig2/shipwreck.jpg" alt="fig1_cont1">|<img src="images/style/shipwreck.jpg">|
|:---:|:---:|
|<img src="images/figures/fig2/starry_night.jpg" alt="fig1_cont1">|<img src="images/style/starry_night.jpg">|
|<img src="images/figures/fig2/scream.jpg" alt="fig1_cont1">|<img src="images/style/scream.jpg">|
|<img src="images/figures/fig2/femme_nue_assise.jpg" alt="fig1_cont1">|<img src="images/style/assise.jpg">|
|<img src="images/figures/fig2/composition.jpg" alt="fig1_cont1">|<img src="images/style/composition.jpg">|


## Future Work
**Definition of Representation.** One advantanges of using neural networks on images is that there already exist perhaps the most useful and direct way to represent an image using numbers - pixel values. But this representation is not necessarily the only way to represent visual content. If there exist a different kind of "embedding" that encodes objects or relationship between pixels in a different way, content and style representation might change the way style transfer model defines the relationship between objects, or even color. 

**Autoencoders and Compression

**CNNs to Other Types of Neural Nets.** One inspiration of Convolutional Neural Networks is the hierachical structure of simple cells and complex cells in the human visual cortex. Layer by layer, using convolution operation, an artifical neuron serves as a computing unit that summaries information from previous layer and compresses into a smaller space, which is then passsed onto the later layers. This type of model is one of many ways of compressing into a more meaningful and less redundant representation. Other type

**Losses and differences.** The current style transfer model utilizes mean square error, which computes the difference between pixel values from the content or style image and the synthsized image. From a mathematical point of view, this seems logical and reasonable. But, a difference in pixel value may not necessarily imply a difference in content or style. For instance, if we were to create a synthsized image that is more invariant to the position of objects in our synthesized image, calculate the exact difference in pixel at each coordinate would not be sensible. In other words, the definition of loss when considering objects may require a much more extensive function than computing losses. 

## Further Readings
		

## References
Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. 2016. “Image Style Transfer Using Convolutional Neural Networks.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). doi:10.1109/cvpr.2016.265.

Gatys, L., Ecker, A. and Bethge, M. (2016). A Neural Algorithm of Artistic Style. Journal of Vision, 16(12), p.326.
