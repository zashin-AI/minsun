import tensorflow as tf

import model

def top_k_logits(logits, k):
    if k == 0:
        # no 절단
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        # tf.newaxis : size(차원) 변경

        return tf.where(
            # tf.where(bool type 텐서, true일 때 출력값, false일 때 출력값)
            # x, y가 없으면 참 요소의 좌표(2D 텐서)를 반환한다. 
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10, # -100억
            # tf.ones_like : 모든 요소가 1로 설정된 tensor와 동일한 유형 및 모양의 tensor를 리턴한다.
            logits,
        )
    
    # tf.cond :  tf.equal이면 logits 동작이 실행되고, _top_k()는 실행되지 않는다.
    return tf.cond( 
       tf.equal(k, 0), # k == 0 
       lambda: logits,
       lambda: _top_k(),
    )



def top_p_logits(logits, p):
    """핵심 sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1) 
    # 내림차순으로 logits을 정렬, sort() = sort(axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    # tf.cumsum : 누적 합계를 수행 (ex. ([a, b, c])   # [a, a + b, a + b + c] )
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


# def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
#     if start_token is None:
#         assert context is not None, 'Specify exactly one of start_token and context!'
#     else:
#         assert context is None, 'Specify exactly one of start_token and context!'
#         context = tf.fill([batch_size, 1], start_token)

#     def step(hparams, tokens, past=None):
#         lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

#         logits = lm_output['logits'][:, :, :hparams.n_vocab]
#         presents = lm_output['present']
#         presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
#         return {
#             'logits': logits,
#             'presents': presents,
#         }

#     with tf.name_scope('sample_sequence'):
#         def body(past, prev, output):
#             next_outputs = step(hparams, prev, past=past)
#             logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
#             logits = top_k_logits(logits, k=top_k)
#             logits = top_p_logits(logits, p=top_p)
#             samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
#             return [
#                 next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
#                 samples,
#                 tf.concat([output, samples], axis=1)
#             ]

#         past, prev, output = body(None, context, context)

#         def cond(*args):
#             return True

#         _, _, tokens = tf.while_loop(
#             cond=cond, body=body,
#             maximum_iterations=length - 1,
#             loop_vars=[
#                 past,
#                 prev,
#                 output
#             ],
#             shape_invariants=[
#                 tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
#                 tf.TensorShape([batch_size, None]),
#                 tf.TensorShape([batch_size, None]),
#             ],
#             back_prop=False,
#         )

#         return tokens
