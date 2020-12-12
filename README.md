# SanGuoGAN

三國志11 Character profile generator via Conditional Wasserstein Generative Adversarial Networks (cWGAN).

This toy project is a cWGAN implementation trained on 790 character profile artworks by 三國志11 (property of Koei Co., Ltd.). The artworks are stylishly similar and can demonstrate the power of deep generative learning. 

The image data are collected from the game directly, and the character stats data are obtain from the summary table made by cws0324@yahoo.com.tw. 

To cleanse the data, we match images with character stats (Leadership, Martial Arts(War), Intelligence, Politics, and Charisma) by the character’s Chinese name. Note that some characters have multiple artworks (young and old), but the stats are the same (sorry 吕蒙). We visualize a random batch of images and stats below. 

[Data Example](examples/dataexample.png)
[Stats Example](examples/dataexamplestats.png)

