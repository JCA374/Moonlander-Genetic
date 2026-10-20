# Lesson 1: The task

## l1_title

Lesson one. The task. Before we evolve anything, let's look closely at the problem our pilot has to solve.

## task

This is Lunar Lander, a small physics game from the Gymnasium library. The lander starts near the top of the screen with a random push, and it has to touch down softly on the pad between the two flags.

## task.2

Many times a second, the lander senses eight numbers. Its position, across and up. Its speed, across and up. Its tilt, and how fast it is spinning. And two switches that turn on when a leg touches the ground.

## task.3

With those eight numbers, it picks one of four actions. Do nothing. Fire the left engine. Fire the main engine underneath. Or fire the right engine. The side engines push the lander sideways and spin it, and the main engine pushes it up.

## sites

Every episode builds a new moon. The mountains are random, and the lander starts with a random shove, so no two flights are the same.

## sites.2

The landing pad never moves, though. It's always in the middle, at the same height, and the lander's position inputs are measured from the pad. Zero, zero means right on target. During training, every pilot is tested on ten random moons, so evolution has to find a rule that works on all of them, instead of memorising one.

## reward

The game keeps score while it flies. Moving toward the pad, slowing down, and staying level all earn a little. Drifting away, speeding up, and tilting all cost a little.

## reward.2

There are bigger events too. Each leg that touches the ground earns ten points. Coming to rest earns a hundred. Crashing costs a hundred.

## reward.3

And fuel isn't free. Every frame the main engine fires costs a third of a point. The side engines cost a tenth as much.

## reward.4

Add it all up over one flight. A score of two hundred or more counts as solved.
