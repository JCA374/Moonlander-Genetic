# Lesson 7: Reading the weights

## l7_title

Lesson seven. Reading the weights. Because the pilot is linear, we can understand exactly what it learned. Let's take it apart.

## w_worked

Start with one decision. The lander is high up and falling fast. Each input is multiplied by its weight, and the products are added up per action. The falling speed gives the main engine a big push, and takes a lot away from do nothing. It's close, but the main engine wins.

## w_worked.2

A moment later, the lander is tilted to the right. The tilt now adds votes to the left engine and takes them from the others. The left engine wins, and turns the lander back upright.

## w_worked.3

Further down, the lander is spinning to the left. Spin has the biggest weight of all, on the right engine, and here it counts for it. The right engine wins, and spins the lander back.

## w_worked.4

And on the ground, with both legs down. The two leg switches each add a big vote for do nothing. Do nothing wins, and the lander stays still. That's the rule the settle penalty taught it.

## w_scale

But be careful. A big weight doesn't always mean a big effect. Here are the sizes of all the weights.

## w_scale.2

What also matters is how much each input actually changes in flight. Altitude and drift move a lot. Spin moves less, so its large weights are multiplied by smaller numbers. The legs are always zero in the air.

## w_scale.3

Multiply the two, and you get the influence of each weight. The ones that matter most are tilt and spin on the right engine, and drift on the left engine.

## w_policy_tilt

We can also draw a rule as a map. Across is the tilt, and up and down is the spin, with every other input held at a typical value. Each colour is the action the pilot would choose there.

## w_policy_tilt.2

The white line is a real flight. The border between the two engines runs diagonally through the middle. Tilt or spin one way, and the pilot fires the engine that turns it back, so it stays upright with small corrections.

## w_policy_fall

Here's a second map. Across is the height, and up and down is the falling speed.

## w_policy_fall.2

Falling fast and low, it fires the main engine. Falling slowly, it steers with the side engines instead. And high up, it doesn't brake yet. It saves the main engine for later. The real flight rides along that border all the way down.

## w_story

Now let's put it together, one action at a time. The main engine is the brake. Falling speed has the biggest weight on it, so the faster the lander falls, the harder it wants to brake.

## w_story.2

Do nothing is the landing rule. Each leg that touches the ground adds a big vote for it. With both legs down, it wins, and the engines go quiet.

## w_story.3

The left engine fixes drift and tilt. It fires when the lander drifts right, sits right of the pad, tilts right, or spins clockwise. It pushes the lander back toward the middle.

## w_story.4

And the right engine is its mirror image. Drifting left, tilting left, or spinning the other way, and the right engine pushes back. Nobody told evolution the lander is symmetric. It found that on its own.

## w_ablation

Finally, let's test which weights the pilot really needs. We set one weight to zero, fly thirty episodes, and see how many still land. Then we put it back, and try the next one.

## w_ablation.2

Most weights can go with little harm. But remove the rule that links falling speed to the main engine, and not a single flight lands. Without it, the pilot never brakes, and simply falls. The leg rules matter almost as much. Remove one of them, and most landings fail.
