"""
Scene lists for the explainer and the lessons. Each video's narration lives in
video_tools/narration/<slug>.md (one `## <scene>[.n]` section per beat) and is voiced with
the voice-text pipeline into voice/<slug>/.
"""

import scenes as S

# Rules for the "what it learned" scene: (action column, cells to outline). Cell row 8 = bias.
STORY_RULES = [(2, [(3, 2)]),                         # main engine: the brake (falling speed)
               (0, [(6, 0), (7, 0)]),                 # do nothing: both legs down -> stay still
               (1, [(2, 1), (0, 1), (4, 1), (5, 1)]), # left engine: drift/sit right, tilt right, spin clockwise
               (3, [(2, 3), (0, 3), (4, 3), (5, 3)])] # right engine: the mirror image

LESSONS = [
    ('lesson-01', 'The task', 'what the lander senses, does, and is scored on'),
    ('lesson-02', 'The controller', '36 numbers that fly a spacecraft'),
    ('lesson-03', 'Population and fitness', 'many candidates, and a way to rank them'),
    ('lesson-04', 'Inside CMA-ES', 'how the search decides where to look next'),
    ('lesson-05', 'Generations', 'watching a population improve'),
    ('lesson-06', 'Evolution in flight', 'the best pilot along the way'),
    ('lesson-07', 'Reading the weights', 'what evolution actually found'),
]


def explainer(ctx):
    return [
        S.scene_title(ctx), S.scene_task(ctx), S.scene_sites(ctx), S.scene_controller(ctx), S.scene_population(ctx),
        S.scene_loop(ctx), S.scene_cmaes(ctx), S.scene_fitness(ctx),
        S.scene_gen_fitness(ctx), S.scene_gen_ghosts(ctx), S.scene_gen_cloud(ctx),
        S.scene_gen_weights(ctx), S.scene_evolution(ctx),
        S.scene_worked(ctx, moments=('falling', 'landed')), S.scene_policy_map(ctx, ry=(-0.4, 0.4)),
        S.scene_story(ctx, STORY_RULES), S.scene_outro(ctx),
    ]


def lesson(ctx, n):
    slug, title, sub = LESSONS[n - 1]
    card = S.title_card(f'l{n}_title', f'Lesson {n}', title, sub)
    body = {
        1: lambda: [S.scene_task(ctx), S.scene_sites(ctx), S.scene_reward(ctx)],
        2: lambda: [S.scene_controller(ctx), S.scene_anatomy(ctx),
                    S.scene_worked(ctx, moments=('falling',))],
        3: lambda: [S.scene_population(ctx),
                    S.scene_fitness(ctx, reveal=((0, 0.1), (1, 0.05), (2, 0.05), (3, 0.05), (4, 0.05)))],
        4: lambda: [S.scene_loop(ctx), S.scene_cmaes_cov(ctx), S.scene_cmaes(ctx)],
        5: lambda: [S.scene_gen_fitness(ctx), S.scene_gen_ghosts(ctx), S.scene_gen_cloud(ctx)],
        6: lambda: [S.scene_evolution(ctx), S.scene_gen_weights(ctx)],
        7: lambda: [S.scene_worked(ctx, moments=('falling', 'tilted', 'spinning', 'landed')), S.scene_scale(ctx),
                    S.scene_policy_map(ctx, name='w_policy_tilt', ry=(-0.4, 0.4)),
                    S.scene_policy_map(ctx, name='w_policy_fall', fx=1, fy=3, rx=(0.0, 1.45),
                                       ry=(-0.8, 0.15), subtitle='height vs. falling speed, '
                                       'everything else held steady'),
                    S.scene_story(ctx, STORY_RULES), S.scene_ablation(ctx)],
    }[n]()
    return slug, [card] + body
