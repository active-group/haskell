-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.Minimize
  ( Minimizer
  , minimizeWith
  , gradientDescent
  , gradientDescentRef
  , OneOfAdamDataTypes
  , AdamConfig(..)
  , adam
  , adam'
  , adamRef
  , adamRef'
  ) where

import Data.Complex (Complex)
import Data.Int (Int8,Int16,Int32,Int64)
import Data.Word (Word8,Word16,Word32,Word64)
import Control.Monad (zipWithM)
import Data.Default (Default(..))
import Data.List (zipWith4)
import Data.Maybe (fromMaybe)

import qualified TensorFlow.Core     as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops      as TF (scalar, mul, zerosLike)
import qualified TensorFlow.Variable as TF

import qualified TensorFlow.Tensor   as TF (Rendered, ToTensor)

import qualified TensorFlow.GenOps.Core as TFO (applyAdam, assignAdd, assign)
import qualified TensorFlow.Ops         as TFO (initializedVariable,
                                                zeroInitializedVariable)

-- | Functions that minimize a loss w.r.t. a set of 'TF.Variable's or 'TF.Tensor TF.Ref's.
--
-- Generally only performs one step of an iterative algorithm.
--
-- 'Minimizer's are defined as a function of the gradients instead of
-- the loss so that users can apply transformations to the gradients.
newtype Minimizer t a m = Minimizer
  { minimize :: (TF.GradientCompatible a, TF.TensorType a, TF.MonadBuild m, TF.ToTensor t, TF.Rendered t) =>
                            [t a] -> [TF.Tensor TF.Value a] -> m TF.ControlNode
  }

minimizer :: forall a m t n. TF.Nodes n => (t a -> TF.Tensor TF.Build a -> m n) -> a -> Minimizer t a m
minimizer assignAdd learningRate =
  Minimizer
    { minimize =
        \params grads ->
          TF.withNameScope "gradientDescent" $ do
            let applyGrad param grad = assignAdd param (TF.scalar (-learningRate) `TF.mul` grad)
            TF.group =<< zipWithM applyGrad params grads
    }

-- | Convenience wrapper around 'TF.gradients' and a 'Minimizer'.
minimizeWith ::
     (TF.MonadBuild m, TF.GradientCompatible a, TF.Rendered t, TF.ToTensor t)
  => Minimizer t a m
  -> TF.Tensor v a -- ^ Loss.
  -> [t a] -- ^ Parameters of the loss function.
  -> m TF.ControlNode
minimizeWith m loss params = TF.gradients loss params >>= minimize m params

-- | Perform one step of the gradient descent algorithm for TF.Variable.
gradientDescent ::
     (TF.MonadBuild m,
     TF.GradientCompatible a)
  => a -- ^ Learning rate.
  -> Minimizer TF.Variable a m
gradientDescent = minimizer TF.assignAdd

-- | Perform one step of the gradient descent algorithm for `TF.Tensor TF.Ref`.
gradientDescentRef ::
    (TF.MonadBuild m,
     TF.GradientCompatible a)
  => a -- ^ Learning rate.
  -> Minimizer (TF.Tensor TF.Ref) a m
gradientDescentRef = minimizer TFO.assignAdd

data AdamConfig t = AdamConfig
    { adamLearningRate :: t
    , adamBeta1        :: t
    , adamBeta2        :: t
    , adamEpsilon      :: t
    }

type OneOfAdamDataTypes t =
        TF.OneOf '[ Complex Double, Complex Float
                  , Int16, Int32, Int64, Int8, Word16, Word32, Word64, Word8
                  , Double, Float] t

instance Fractional t => Default (AdamConfig t) where
  -- Recommended defaults from the adam paper.
  def = AdamConfig 0.001 0.9 0.999 1e-8

-- | Perform one step of the adam algorithm for `TF.Variable`.
--
-- See https://arxiv.org/abs/1412.6980.
--
-- NOTE: Currently requires all 'TF.Variable's to have an 'TF.initializedValue'.
adam :: (OneOfAdamDataTypes t, Fractional t) => Minimizer TF.Variable t TF.Build
adam = adam' def

adam' :: (OneOfAdamDataTypes t, Fractional t) => AdamConfig t -> Minimizer TF.Variable t TF.Build
adam' config =
  let errorMsg = "TensorFlow.Minimize.adam requires an initial value for all variables"
      initVal = fromMaybe (error errorMsg) . TF.initializedValue
   in adam''
        config
        (mapM (TF.initializedVariable . TF.zerosLike . initVal))
        TF.initializedVariable
        TF.resourceApplyAdam
        TF.readValue
        TF.assign

adamRef :: (OneOfAdamDataTypes t, Fractional t) => [TF.Shape] -> Minimizer (TF.Tensor TF.Ref) t TF.Build
adamRef = adamRef' def

-- | Perform one step of the adam algorithm for `TF.Tensor TF.Ref`.
-- |
--   Similar solution as for `TF.Variable` works sometimes...
--   Creating initialized variables the same as for `TF.Variable` is `(TFO.initializedVariable . TF.zerosLike . TF.value)`
--   but gives many times runtime error: "attempting to use uninitialized value variable"
adamRef' :: (OneOfAdamDataTypes t, Fractional t) => AdamConfig t -> [TF.Shape] -> Minimizer (TF.Tensor TF.Ref) t TF.Build
adamRef' config shapes =
  adam''
    config
    (\_ -> mapM TFO.zeroInitializedVariable shapes)
    TFO.initializedVariable
    TFO.applyAdam
    TF.expr
    TFO.assign

adam'' :: forall t a n . (TF.Nodes n, TF.ToTensor t, TF.Rendered t, OneOfAdamDataTypes a, Fractional a) =>
     AdamConfig a
  -> ([t a] -> TF.Build [t a])
  -> (TF.Tensor TF.Build a -> TF.Build (t a))
  -> (t a -> t a -> t a -> TF.Tensor TF.Build a -> TF.Tensor TF.Build a -> TF.Tensor TF.Build a -> TF.Tensor TF.Build a -> TF.Tensor TF.Build a -> TF.Tensor TF.Build a -> TF.Tensor TF.Value a -> TF.Build n)
  -> (t a -> TF.Tensor TF.Build a)
  -> (t a -> TF.Tensor TF.Build a -> TF.Build n)
  -> Minimizer t a TF.Build
adam'' config initVarZero initVar applyAdam readValue assign = Minimizer
  { minimize = \params grads -> TF.withNameScope "adam" $ do
    let lr = TF.scalar (adamLearningRate config)
        beta1 = TF.scalar (adamBeta1 config)
        beta2 = TF.scalar (adamBeta2 config)
        epsilon = TF.scalar (adamEpsilon config)
    -- Create adam state variables.
    ms <- initVarZero params
    vs <- initVarZero params
    beta1Power <- initVar beta1
    beta2Power <- initVar beta2
    -- Perform adam update.
    let applyGrad param m v = applyAdam param m v
                                 (readValue beta1Power)
                                 (readValue beta2Power)
                                 lr beta1 beta2 epsilon
    updateVars <- sequence $ zipWith4 applyGrad params ms vs grads
    -- Update beta variables after adam update.
    let updateBeta betaPower beta =
            TF.withControlDependencies updateVars
                (assign betaPower (readValue betaPower `TF.mul` beta))
    updateBeta1 <- updateBeta beta1Power beta1
    updateBeta2 <- updateBeta beta2Power beta2
    TF.group (updateBeta1:updateBeta2:updateVars)
  }
