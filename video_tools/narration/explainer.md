# Moonlander-Genetic explainer

## title

What if you could teach a spacecraft to land on the Moon with just thirty-six numbers? No neural network, and no gradients. Only evolution.

## task

This is Lunar Lander. The goal is to touch down softly between the two flags. Gravity pulls the lander down, and every burst of an engine costs fuel.

## task.2

At every step, the lander senses eight numbers. Where it is, how fast it is moving, how it is tilted and spinning, and whether each leg touches the ground.

## task.3

And at every step, it chooses one of four actions. Do nothing, fire the left engine, fire the main engine, or fire the right engine.

## sites

Every flight happens on a new moon. The mountains are random, and the lander gets a random shove at the start. So the pilot can't memorise one flight.

## sites.2

But the landing pad is always in the middle, at the same height. The lander measures its position from the pad, so the goal is always the same: get to zero, zero. Each pilot is tested on ten random moons, so evolution has to find a rule that works everywhere.

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

Then we add a big bonus for landing, and a small one for landing quickly. And one penalty: every engine burst after both legs are down costs half a point. A good pilot lands, and then stays still.

## gen_fitness

Now let's watch it learn. Each dot is one pilot, and each column is one generation of fifty pilots.

## gen_fitness.2

In the first twenty generations, nearly every pilot still crashes. But the dots keep creeping upward, because each generation is sampled around the best of the last one.

## gen_fitness.3

Then the whole population lifts. By generation fifty, more than half the pilots land. By seventy-five, almost all of them do. The orange line is the best pilot, the blue line is the average, and the gap closes as the population settles on one strategy.

## gen_ghosts

Here is the same story in flight. All fifty pilots of one generation start from exactly the same spot, and we draw the path of each one.

## gen_ghosts.2

Generation zero is chaos. Pilots fall straight down, or fly off the screen. Five generations later, not one of them lands.

## gen_ghosts.3

By generation twenty, three pilots make it. By thirty, fourteen. By forty, twenty-seven. And by fifty, thirty-six of the fifty pilots land.

## gen_ghosts.4

By the end, all fifty land. The whole population flies almost the same path, a smooth curve down to the middle of the pad.

## gen_cloud

This is the population as points. Each pilot is thirty-six numbers, flattened down to two, so each dot is a whole pilot.

## gen_cloud.2

The centre of the cloud travels a long way from where it started, toward the region where pilots land. The white line is its path.

## gen_cloud.3

And there's a neat trick here. Only the highest score matters, so making every weight bigger doesn't change any decision. The weights grow about a hundred times bigger, while the step size stays around one. So each step gets smaller and smaller compared with the weights, and the search turns into fine-tuning.

## gen_weights

Here is the centre of the cloud, drawn as the weight matrix itself. At generation zero it is noise. Watch the pattern appear.

## gen_weights.2

The strongest rules are about tilt, spin and drift. Exactly the things a human pilot would watch.

## evolution

Here are the best pilots from different points in training, all facing the same start.

## evolution.2

The random pilot flies away, and the champion of generation ten crashes. From generation thirty on, the best pilot lands, and each one lands a little more cleanly.

## w_worked

Because the pilot is so simple, we can follow every decision. Here the lander is falling fast. Each input times its weight adds to each score. The falling speed pushes the main engine up and do nothing down, and the main engine wins by a nose.

## w_worked.2

And here it has just landed, with both legs on the ground. The leg switches add big votes to do nothing, so do nothing wins, and the lander stays still. That's the rule the settle penalty taught it.

## w_policy_map

We can draw a whole rule at once. Across is the tilt, and up and down is the spin. Each colour is the action the pilot would choose.

## w_policy_map.2

The white line is a real flight. Tilt or spin one way, and it fires the engine that turns it back. The border between them runs right through the middle, so the pilot keeps itself upright with small corrections.

## w_story

So what did evolution find? The main engine is the brake. Falling speed has the biggest weight on it, so the faster the lander falls, the harder it wants to brake.

## w_story.2

Do nothing is the landing rule. Each leg that touches the ground adds a big vote for it. With both legs down, it wins, and the engines go quiet.

## w_story.3

The left engine fixes drift and tilt. It fires when the lander drifts right, sits right of the pad, tilts right, or spins clockwise. And it pushes the lander back.

## w_story.4

And the right engine is its mirror image. Drifting left, tilting left, or spinning the other way, and the right engine pushes back. Evolution found the symmetry on its own.

## outro

Tested on one hundred new landings it never saw in training, this pilot lands ninety-five times. And once it's down, it stays down.

## outro.2

All of it from thirty-six numbers and a loop of trial and error. The code is on GitHub. Run it, and watch your own pilot evolve.
