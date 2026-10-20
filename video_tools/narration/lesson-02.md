# Lesson 2: The controller

## l2_title

Lesson two. The controller. How can thirty-six numbers fly a spacecraft?

## controller

Our pilot is one small table of numbers, called a matrix. Each of the eight inputs is multiplied by four weights, one for each action. Then a bias is added to each total.

## controller.2

That gives four scores. Whichever score is highest, that action fires. Watch the scores on the right change as the lander flies, and the winning action change with them.

## controller.3

Eight inputs times four actions is thirty-two weights. Add four biases, and you have thirty-six numbers. That is the entire pilot.

## w_anatomy

Let's learn to read the table. Each row is one input, and each column is one action. Orange cells are positive weights, and blue cells are negative. The brighter the cell, the bigger the number.

## w_anatomy.2

Read across a row, and you see how one input votes. Horizontal speed pushes some actions up and others down, all at the same time.

## w_anatomy.3

Read down a column, and you see everything that argues for one action. This is the left engine's column.

## w_anatomy.4

A single cell is a single rule. A positive weight means the bigger this input, the more I want this action. A negative weight means the opposite. Here: drifting right is a strong vote for the left engine.

## w_anatomy.5

The last row is the bias. It's a head start, or a handicap, that each action gets, whatever the state.

## w_worked

Now let's watch one real decision. Each input is multiplied by its weight, and the products are added up in each column. The lander is falling fast, so the falling speed pushes the main engine's score up. It has the highest total, so the main engine fires.
