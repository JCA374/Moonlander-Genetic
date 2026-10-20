# Moonlander-Genetic explainer

## title

What if you could teach a spacecraft to land on the Moon with just thirty-six numbers? No neural network, and no gradients. Only evolution.

## task

This is Lunar Lander. The goal is to touch down softly between the two flags. Gravity pulls the lander down, and every burst of an engine costs fuel.

## task.2

At every step, the lander senses eight numbers. Where it is, how fast it is moving, how it is tilted and spinning, and whether each leg touches the ground.

## task.3

And at every step, it chooses one of four actions. Do nothing, fire the left engine, fire the main engine, or fire the right engine.

## controller

The pilot is one small table of numbers. Each of the eight inputs is multiplied by a weight for each action, and a bias is added. That gives four scores.

## controller.2

The action with the highest score fires. Watch the scores rise and fall as the lander flies.

## controller.3

Eight times four weights, plus four biases, makes thirty-six numbers. Finding good values for them is the job of the genetic algorithm.

## population

We start with a population of random pilots. Each one is just thirty-six random numbers.

## population.2

Every pilot flies, and gets a fitness score. Random pilots crash. Some of them crash a little less badly than others.

## population.3

The best half are kept. They decide where the next generation comes from.

## loop

That's the whole loop. Sample a population, fly every pilot, rank them, and move toward the winners. Then repeat. Nothing is differentiated. The algorithm only ever sees the scores.

## cmaes

The algorithm we use is called CMA-ES. Here it is on a toy problem with only two numbers, so we can watch it work. Dark means good, and bright means bad.

## cmaes.2

It keeps a cloud of candidates, shaped like a stretched bell curve. Each generation it samples new points, keeps the best half, and moves the centre toward them.

## cmaes.3

It also learns the shape of the problem. The cloud stretches along the valley, races down it, and shrinks as it closes in on the answer.

## fitness

How do we score a pilot? It flies ten episodes, and we start from its average reward in the game. That already punishes crashing and wasted fuel.

## fitness.2

Then we add a big bonus for landing, and a small one for landing quickly.

## gen_fitness

Now let's watch it learn. Each dot is one pilot, and each column is one generation of fifty pilots.

## gen_fitness.2

In the first thirty generations, nearly every pilot still crashes. But the dots keep creeping upward, because each generation is sampled around the best of the last one.

## gen_fitness.3

Then the whole population lifts. By generation seventy-five, most pilots land. The orange line is the best pilot, the blue line is the average, and the gap between them closes as the population settles on one strategy.

## gen_ghosts

Here is the same story in flight. All fifty pilots of one generation start from exactly the same spot, and we draw the path of each one.

## gen_ghosts.2

Generation zero is chaos. Pilots fall straight down, or fly off the screen. Five generations later, not one of them lands yet.

## gen_ghosts.3

By generation ten, a few paths bend toward the pad. By forty, more of them arrive. And by fifty, nineteen of the fifty pilots land.

## gen_ghosts.4

By generation one hundred, forty-eight of them land. The whole population flies almost the same path, a smooth curve down between the flags.

## gen_cloud

This is the population as points. Each pilot is thirty-six numbers, flattened down to two, so each dot is a whole pilot.

## gen_cloud.2

The centre of the cloud travels a long way from where it started, toward the region where pilots land. The white line is its path.

## gen_cloud.3

And there's a neat trick here. Only the highest score matters, so making every weight bigger doesn't change any decision. As the weights grow, the same step size becomes a smaller and smaller step, and the search turns into fine-tuning.

## gen_weights

Here is the centre of the cloud, drawn as the weight matrix itself. At generation zero it is noise. Watch the pattern appear.

## gen_weights.2

The strongest rules are about speed. Drifting sideways, falling, and spinning. They are exactly the things a human pilot would watch.

## evolution

Here are the best pilots from different points in training, all facing the same start.

## evolution.2

The random pilot flies away, and the early champion crashes. By generation thirty the best pilot lands, and later ones land more cleanly.

## w_worked

Because the pilot is so simple, we can follow every decision. Here the lander is falling fast. Each input times its weight adds to each score. The falling speed pushes the main engine up and do nothing down, so the main engine wins.

## w_worked.2

A moment earlier, it was tilted to the right. Now the left engine collects just enough votes to win, and it turns the lander back upright.

## w_policy_map

We can draw a whole rule at once. Across is the tilt, and up and down is the spin. Each colour is the action the pilot would choose.

## w_policy_map.2

The white line is a real flight. The pilot stays near the middle, hopping between the engines to keep itself upright.

## w_story

So what did evolution find? The main engine fires when the lander is falling. Its bias is negative, so it stays off unless the fall is fast enough.

## w_story.2

Do nothing gets a strong vote when the lander is rising, and a strong vote against it when the left leg is down. This pilot almost never just waits.

## w_story.3

The left engine fixes drift and spin. It fires when the lander drifts right or spins clockwise, and it pushes the lander back.

## w_story.4

And the right engine is the default. It fires almost half the time, with small, cheap pulses whenever no other rule wins. When the lander drifts left, the left engine loses its votes, and the right engine takes over.

## outro

Tested on one hundred new landings it never saw in training, this pilot lands ninety-six times.

## outro.2

All of it from thirty-six numbers and a loop of trial and error. The code is on GitHub. Run it, and watch your own pilot evolve.
