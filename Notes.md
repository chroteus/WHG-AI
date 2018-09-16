### [Task 0 - Get data]

> ##### "I need to get data from game as an array I can feed into neural network. Plus, I need to read off the score."

* Added second screen and took screenshots of it to get game data
Used OCR to read level and death number
* Along the way realized, both monitors run off the same x11 server, so I wouldn't be able to use computer while AI is running.
---
> ##### "Oh, I know! I'll use a VM!"

* So, I set up a VM using VBox with Bodhi Linux.  
* I install Chromium on it, since it can use Pepper Flash and set it to autostart the .swf file for World's Hardest Game.  
* At first, I tried to use VBox Python API to take screenshot for game data, but it's broken for screenshots! :frowning:

---
> ##### "How else can I access my VM? VNC?"

* I setup a VNC server on guest and got game data using a Python VNC library called "vncdotool". Luckily, it worked fine and was fast enough.
* I still used OCR to read deaths and level.
---

### [Task 1 - Objective]

> ##### "We cannot reward the agent unless we can quantify how well it is doing."

> ##### "Using a human for rating purposes is infeasible as the human is lazy, so we'll resort to using a neural net which will look at the screen and output a number 0-1 which will tell us how well our agent is performing."

> ##### "It HAS to be accurate and well-crafted or else our agent will be misguided when learning."


* __[RUN 1-3]__ **[2/10]**  -> Bad.
    * I generated small amounts of data and trained the net. The result was terrible. I'm sure there wasn't enough data and I didn't do any validation.

* __[Run 4 and 5]__ **[2/10]** -> Bad.
    * Increased data, started using validation and trained.  
    * Result was still bad. On #5, changed loss to BCE, just to see result. Didn't improve performance at all.


* __[Run 6]__ **[3/10]** -> Bad.
    * Increased input image from 96x56 to 144x84 (x1.5). Regenerated train data with new resolution, on only the 1st level to save time.

    * Result is better, it can kinda recognize when player passes half the screen, but it regards enemies as important when quantifying position.

* __[Run 7]__ **[0/10]** -> Doesn't even converge.
    * For experiment, remove convolutional layers, reduce image size back to 96x56, and use linear layers ONLY (no convolution!)
    * Result = Does not converge.

* __[Run 8]__ **[4/10]** -> Bad.
    * Most complicated net used out of all ValueNets.
    * Result = Just barely better than previous runs. :joy:
---
#### [!!! Bug !!!]
* Discovered an embarrassingly stupid bug which made %75 of data not being used. There was an error in batching which filled all of the mini-batch with its first element.  
No wonder I got these awful results.



#### [Insight!]
###### Along with the bug, I've discovered something else.

* Value output by net is somewhat affected by position of blue critters, which we don't care about (it's the job of the agent to dodge them). Hence even if player doesn't move, value output by ValueNet varies due to the noise of blue critters.

* What if we simply use mean of last 5-10 values recorded instead of just value at that exact moment?
* Result: It works MUCH better. When averaged out, noise somewhat cancels out itself.
---
* __[Runs 9 and 10]__ **[8/10]** -> Almost there
    * Fixing the bug had a HUGE effect on performance. It's SO, SO much better now.
       However, it still needs to be averaged out as blue critters are still causing noise.

* ###### __[Runs 11 and 12 not important]__

* __[Runs 13 - 16]__ **[9/10]** -> Almost perfect
    * Made net architecture a bit more complex, which had a very positive effect on accuracy.  
    *HOWEVER:* I've found a blind spot which rewarded player too much at a certain spot on start. So we can't use it as agent will go to that point instead of finish.

* __[Runs 17 and 18]__ **[10/10]** -> Perfect.
    * Fixed blind spot by throwing more data at net. ValueNet can now be used.
        There's still noise picked up due to blue critters, but it's not as pronounced and
        averaging trick works just fine.
    * I'm going to use both #17 and #18 as an ensemble.

---
#### [Conclusion of Task 1]
* Batching bug severely affected my productivity, and I almost gave up on using neural nets for objective, but I've finally made it.

* Judging by timestamps on videos, it took me a week to finally create ValueNet. Admittedly, it felt like a longer time.

---
### [Task 2 - Learning!]

#### Policy training
