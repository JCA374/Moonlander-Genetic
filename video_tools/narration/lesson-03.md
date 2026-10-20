# Lesson 3: Population and fitness

## l3_title

Lesson three. Population and fitness. Evolution needs two things. Many candidates, and a way to tell the good ones from the bad.

## population

We start with a population of pilots. Each one is thirty-six random numbers. In training there are fifty of them. Here are eighteen.

## population.2

Each pilot flies, and we score it. With random numbers, every one of them crashes. But some crash later, or closer to the pad, and that difference is all evolution needs.

## population.3

We keep the best half, and the rest are thrown away. The survivors decide where the next generation will be sampled.

## fitness

So what is the score? Each pilot flies ten different episodes, each with its own terrain and starting push. Then we combine three things.

## fitness.2

First, the average reward from the game itself. That already rewards approaching the pad, and it punishes crashing and wasted fuel.

## fitness.3

Second, a bonus of a hundred points, times the share of episodes that end in a landing. Landing is what we really want, so it's worth a lot.

## fitness.4

Third, a small speed bonus, up to thirty points, for landing quickly. It is multiplied by the landing rate, so a pilot can't earn it by crashing fast.
