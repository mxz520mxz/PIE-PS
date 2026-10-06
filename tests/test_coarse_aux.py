import io
import pickle
import tempfile
import unittest
from pathlib import Path
import torch
from pieps.runtime import coarse_aux_loss, final_normal_loss, build_model, OBJECTIVE
from pieps.training import TRAINER
from pieps.data import load_scene, model_inputs
from test_contracts import example_scene

class CoarseAux(unittest.TestCase):
    def test_zero_weight_and_additive_gradients(self):
        torch.manual_seed(9)
        p=torch.randn(9,3,requires_grad=True)
        c=torch.randn(9,3,requires_grad=True)
        target=torch.randn(9,3)
        valid=torch.tensor([True]*8+[False])
        loss,final,coarse=coarse_aux_loss(p,c,target,valid,.3)
        gf=torch.autograd.grad(final,p,retain_graph=True)[0]
        gc=torch.autograd.grad(coarse,c,retain_graph=True)[0]
        gp,gq=torch.autograd.grad(loss,(p,c),retain_graph=True)
        torch.testing.assert_close(gp,gf)
        torch.testing.assert_close(gq,.3*gc)
        zero,_,_=coarse_aux_loss(p,c,target,valid,0.)
        self.assertTrue(torch.equal(zero,final))
        zp,zc=torch.autograd.grad(zero,(p,c))
        torch.testing.assert_close(zp,gf)
        self.assertEqual(float(zc.abs().sum()),0.)
        self.assertEqual(float(gp[-1].abs().sum()),0.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_coarse_reaches_head_encoder_and_checkpoint_reload(self):
        torch.manual_seed(42)
        model=build_model({},torch.device('cuda:0'))
        model.checkpoint_scorer=True
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scene.pkl'
            path.write_bytes(pickle.dumps(example_scene()))
            batch=load_scene(path,require_target=True)
            p,aux=model(*model_inputs(batch,'cuda:0'),return_aux=True)
            total,final,coarse=coarse_aux_loss(p,aux['coarse_normal'],batch['pixel_n_gt'].cuda(),batch['pixel_valid'].cuda(),.3)
            params=[model.normal_head.weight] if hasattr(model.normal_head,'weight') else list(model.normal_head.parameters())
            params+=list(model.conv_block1.parameters())
            grad=torch.autograd.grad(coarse,params,retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in grad))
            self.assertGreater(sum(float(g.abs().sum()) for g in grad),0)
            opt=torch.optim.Adam(model.parameters(),lr=1e-4)
            total.backward(); opt.step()
            buffer=io.BytesIO()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),objective=OBJECTIVE,trainer=TRAINER,coarse_loss_weight=.3),buffer)
            buffer.seek(0); state=torch.load(buffer,weights_only=False)
            self.assertEqual(state['objective'],'final_plus_coarse_normal_angular_degrees')
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            self.assertTrue(all(torch.isfinite(v).all() for v in model.parameters()))
