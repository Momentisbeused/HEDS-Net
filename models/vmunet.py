from .vmamba import VSSM
import torch
from torch import nn

class VMUNet (nn .Module ):

    def __init__ (self ,input_channels =3 ,num_classes =1 ,depths =[2 ,2 ,9 ,2 ],depths_decoder =[2 ,9 ,2 ,2 ],drop_path_rate =0.2 ,load_ckpt_path =None ,use_axis_bridge =True ,use_deep_supervision =True ,use_coordinate_attention =True ,use_hvst =False ):
        super ().__init__ ()
        self .load_ckpt_path =load_ckpt_path
        self .num_classes =num_classes
        self .input_channels =input_channels
        self .use_hvst =use_hvst
        self .vmunet =VSSM (in_chans =input_channels ,num_classes =num_classes ,depths =depths ,depths_decoder =depths_decoder ,drop_path_rate =drop_path_rate ,use_axis_bridge =use_axis_bridge ,use_deep_supervision =use_deep_supervision ,use_coordinate_attention =use_coordinate_attention ,use_hvst =use_hvst )

    def forward (self ,x ):
        if x .size (1 )==1 and self .input_channels ==3 :
            x =x .repeat (1 ,3 ,1 ,1 )
        output =self .vmunet (x )
        if isinstance (output ,tuple ):
            logits ,ds_features =output
            return (logits ,ds_features )
        return output

    def load_from (self ):
        if self .load_ckpt_path is not None :
            ckpt =torch .load (self .load_ckpt_path ,map_location ='cpu')
            pretrained_dict =ckpt ['model']
            if self .use_hvst :
                mapped_dict ={}
                for k ,v in pretrained_dict .items ():
                    if '.self_attention.'in k :
                        k =k .replace ('.self_attention.','.vss_branch.')
                    if '.ln_1.'in k :
                        k =k .replace ('.ln_1.','.norm1.')
                    mapped_dict [k ]=v
                pretrained_dict =mapped_dict
            model_dict =self .vmunet .state_dict ()
            new_dict ={k :v for k ,v in pretrained_dict .items ()if k in model_dict }
            model_dict .update (new_dict )
            self .vmunet .load_state_dict (model_dict )
            enc_skip =len (pretrained_dict )-len (new_dict )
            print (f'Encoder: loaded {len (new_dict )}/{len (model_dict )} keys; skipped pretrained keys: {enc_skip }')
            model_dict =self .vmunet .state_dict ()
            pretrained_dec ={}
            for k ,v in ckpt ['model'].items ():
                if 'layers.0'in k :
                    pretrained_dec [k .replace ('layers.0','layers_up.3')]=v
                elif 'layers.1'in k :
                    pretrained_dec [k .replace ('layers.1','layers_up.2')]=v
                elif 'layers.2'in k :
                    pretrained_dec [k .replace ('layers.2','layers_up.1')]=v
                elif 'layers.3'in k :
                    pretrained_dec [k .replace ('layers.3','layers_up.0')]=v
            new_dec ={k :v for k ,v in pretrained_dec .items ()if k in model_dict }
            model_dict .update (new_dec )
            self .vmunet .load_state_dict (model_dict )
            dec_skip =len (pretrained_dec )-len (new_dec )
            print (f'Decoder: loaded {len (new_dec )}/{len (model_dict )} keys; skipped pretrained keys: {dec_skip }')
