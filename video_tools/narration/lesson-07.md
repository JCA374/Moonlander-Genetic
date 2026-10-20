# Lesson 7: Reading the weights

## l7_title

Lesson seven. Reading the weights. Because the pilot is linear, we can understand exactly what it learned. Let's take it apart.

## w_worked

Start with one decision. The lander is high up and falling fast. Each input is multiplied by its weight, and the products are added up per action. The falling speed alone gives the main engine a big push, and takes a lot away from do nothing. The main engine wins.

## w_worked.2

A few frames earlier, the lander was tilted to the right. Now it's close. The tilt and the height add votes to the left engine, and it wins by a small margin. Firing it turns the lander back upright.

## w_worked.3

Near the ground, the lander is spinning to the left. Spin has the biggest weight of all, on the left engine, and here it counts against it. The left engine drops far behind, and the right engine wins. That spins the lander back.

## w_scale

But be careful. A big weight doesn't always mean a big effect. Here are the sizes of all the weights.

## w_scale.2

What also matters is how much each input actually changes in flight. Altitude moves a lot. Spin barely moves at all, so its large weights are multiplied by small numbers. The legs are always zero in the air.

## w_scale.3

Multiply the two, and you get the influence of each weight. The three that matter most are drift on the left engine, and falling speed on do nothing and on the main engine.

## w_policy_tilt

We can also draw a rule as a map. Across is the tilt, and up and down is the spin, with every other input held at a typical value. Each colour is the action the pilot would choose there.

## w_policy_tilt.2

The white line is a real flight. It stays near the middle, where the colours meet, so the pilot hops between engines with every small change.

## w_policy_fall

Here's a second map. Across is the height, and up and down is the falling speed.

## w_policy_fall.2

Falling fast, it fires the main engine. Falling slowly while still high up, it uses the side engines. And close to the ground, moving slowly, it does nothing. The real flight zig-zags along the border. The pilot brakes just enough, again and again, all the way down.

## w_story

Now let's put it together, one action at a time. The main engine fires when the lander is falling. Its bias is negative, so it stays off unless the fall is fast enough.

## w_story.2

Do nothing gets a strong vote when the lander is rising, and a strong vote against it when the left leg is down. This pilot almost never just waits.

## w_story.3

The left engine fixes drift and spin. It fires when the lander drifts right, or spins clockwise, and it pushes the lander back.

## w_story.4

And the right engine is the default. It fires almost half the time, in small, cheap pulses, whenever no other rule wins. There's no rule for drifting left. When that happens, the left engine loses its votes, and the right engine wins by default.

## w_ablation

Finally, let's test which weights the pilot really needs. We set one weight to zero, fly thirty episodes, and see how many still land. Then we put it back, and try the next one.

## w_ablation.2

Most weights can go with little harm. But remove the rule that links falling speed to the main engine, and not a single flight lands. Without it, the pilot never brakes, and simply falls.
