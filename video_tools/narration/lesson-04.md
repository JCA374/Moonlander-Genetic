# Lesson 4: Inside CMA-ES

## l4_title

Lesson four. Inside CMA-ES. How does the algorithm decide where to look next?

## loop

Every generation runs the same loop. Sample new pilots. Fly them. Rank them. Then update, and repeat. The algorithm never looks inside the pilot. It only sees the scores.

## cmaes_cov

The new pilots are drawn from a cloud, a bell-shaped distribution over all thirty-six numbers. It has three parts.

## cmaes_cov.2

The mean is the centre of the cloud. It's the algorithm's current best guess for a good pilot.

## cmaes_cov.3

The step size, sigma, sets how far from the mean new candidates land. Big steps explore. Small steps fine-tune.

## cmaes_cov.4

And the covariance sets the shape. It lets the cloud stretch in the directions where good pilots were found, and stay narrow everywhere else.

## cmaes_cov.5

In our problem the cloud has thirty-six dimensions, one per weight. To watch it work, let's shrink it down to two.

## cmaes

Here is a toy problem with two numbers. Dark means low cost, which is good, and bright means high cost. The green cross is the best answer, but the algorithm doesn't know where it is.

## cmaes.2

Each generation, it samples twelve points and keeps the best six, shown in orange. It moves the mean toward them, and reshapes the ellipse to follow the direction the winners came from.

## cmaes.3

Watch the cloud stretch along the valley and race down it. As it closes in, the step size shrinks, until it settles on the answer. Learning that shape is what the name means: covariance matrix adaptation.
