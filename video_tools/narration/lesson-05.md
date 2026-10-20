# Lesson 5: Generations

## l5_title

Lesson five. Generations. Now let's watch the population improve, one generation at a time.

## gen_fitness

Each dot is one pilot. Each column is one generation of fifty pilots, and the height of a dot is its fitness. Red pilots mostly crash, and green ones mostly land.

## gen_fitness.2

In the first thirty generations, nearly every pilot crashes. But each generation is sampled around the best of the last one, so the whole column creeps upward. The best pilot lands half its episodes by generation thirty-two.

## gen_fitness.3

Then the whole population climbs. By generation fifty, a fifth of the pilots land. By seventy-five, most of them do. The orange line is the best pilot, and the blue line is the average. As the population agrees on one strategy, the average catches up with the best.

## gen_ghosts

Fitness is just a number, so let's watch the flights. Every pilot of a generation starts from exactly the same spot, with the same push, and we draw where each one goes.

## gen_ghosts.2

Generation zero is chaos. Pilots fall straight down, or fire an engine and fly off the screen. One generation later, and then five, the paths change shape, but not one of them lands yet.

## gen_ghosts.3

By generation ten, a few paths bend toward the pad. Through twenty and thirty, the paths tighten, but only a handful land. By forty, eight land. And by fifty, nineteen of the fifty pilots land.

## gen_ghosts.4

By seventy-five, forty land. By one hundred, forty-eight. In the end, the trails bundle into one smooth curve between the flags.

## gen_cloud

This is the same population, seen as points. Each pilot is thirty-six numbers. A method called principal component analysis flattens them down to two, so each dot is a whole pilot.

## gen_cloud.2

The centre of the cloud, in blue, travels a long way from where it started, toward the region where pilots land. The white line is its path.

## gen_cloud.3

Here is a neat trick. Only the highest score matters, so if you double every weight, every decision stays the same. As the weights grow, the same step size becomes a smaller and smaller step relative to them. The search turns into fine-tuning on its own.
