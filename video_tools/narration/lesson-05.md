# Lesson 5: Generations

## l5_title

Lesson five. Generations. Now let's watch the population improve, one generation at a time.

## gen_fitness

Each dot is one pilot. Each column is one generation of fifty pilots, and the height of a dot is its fitness. Red pilots mostly crash, and green ones mostly land.

## gen_fitness.2

In the first twenty generations, nearly every pilot crashes. But each generation is sampled around the best of the last one, so the whole column creeps upward. The best pilot lands half its episodes by generation twenty.

## gen_fitness.3

Then the whole population climbs. By generation fifty, more than half the pilots land. By seventy-five, almost all of them do, and by one hundred, every single one. The orange line is the best pilot, and the blue line is the average. As the population agrees on one strategy, the average catches up with the best.

## gen_ghosts

Fitness is just a number, so let's watch the flights. Every pilot of a generation starts from exactly the same spot, with the same push, and we draw where each one goes.

## gen_ghosts.2

Generation zero is chaos. Pilots fall straight down, or fire an engine and fly off the screen. One generation later, one pilot lands, almost by luck. By generation five, none do.

## gen_ghosts.3

At generation ten, still none. Then it takes off. Three pilots land at generation twenty, fourteen at thirty, twenty-seven at forty, and thirty-six at fifty.

## gen_ghosts.4

By seventy-five, forty-eight land. By the end, all fifty, and the trails bundle into one smooth curve down to the middle of the pad.

## gen_cloud

This is the same population, seen as points. Each pilot is thirty-six numbers. A method called principal component analysis flattens them down to two, so each dot is a whole pilot.

## gen_cloud.2

The centre of the cloud, in blue, travels a long way from where it started, toward the region where pilots land. The white line is its path.

## gen_cloud.3

Here is a neat trick. Only the highest score matters, so if you double every weight, every decision stays the same. The weights grew about a hundred times bigger, while the step size stayed around one. So each step got smaller and smaller compared with the weights, and the search turned into fine-tuning on its own.
