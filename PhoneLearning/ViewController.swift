//
//  ViewController.swift
//  PhoneLearning
//
//  Created by Free Debreuil on 2016-04-20.
//  Copyright © 2016 Free Debreuil. All rights reserved.
//

import UIKit
import Metal

class ViewController: UIViewController
{

    override func viewDidLoad()
    {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        initMetal()
//        initGame()
    }
    func initGame()
    {
        // prepare input data
        var myvector = [Float](count: 123456, repeatedValue: 0)
        for (index, value) in myvector.enumerate()
        {
            myvector[index] = Float(index)
        }
        
        // prepare sigmoid function to run on gpu
        // a. initialize Metal
        let (device, commandQueue, defaultLibrary, commandBuffer, computeCommandEncoder) = initMetal()
        
        // b. set up a compute pipeline with Sigmoid function and add it to encoder
        let sigmoidProgram = defaultLibrary.newFunctionWithName("sigmoid")
//        var pipelineErrors = NSErrorPointer()
//        var commandBuffer = MTLCommandBuffer();
        let computePipelineFilter = try? device.newComputePipelineStateWithFunction(sigmoidProgram!)//, completionHandler: rror: pipelineErrors)
        computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
        
        // create gpu input and output data
        // a. calculate byte length of input data - myvector
        let myvectorByteLength = myvector.count*sizeofValue(myvector[0])
        
        // b. create a MTLBuffer - input data that the GPU and Metal and produce
        let inVectorBuffer = device.newBufferWithBytes(&myvector, length: myvectorByteLength, options: MTLResourceOptions.CPUCacheModeDefaultCache)
        
        // c. set the input vector for the Sigmoid() function, e.g. inVector
        //    atIndex: 0 here corresponds to buffer(0) in the Sigmoid function
        computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, atIndex: 0)
        
        // d. create the output vector for the Sigmoid() function, e.g. outVector
        //    atIndex: 1 here corresponds to buffer(1) in the Sigmoid function
        var resultdata = [Float](count:myvector.count, repeatedValue: 0)
        let outVectorBuffer = device.newBufferWithBytes(&resultdata, length: myvectorByteLength, options: MTLResourceOptions.CPUCacheModeDefaultCache)
        computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, atIndex: 1)
        
        // end encoding for the compute command buffer
        computeCommandEncoder.endEncoding();
        
        // configure gpu threads
        // hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
        let threadsPerGroup = MTLSize(width:32,height:1,depth:1)
        let numThreadgroups = MTLSize(width:(myvector.count+31)/32, height:1, depth:1)
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
    }
    func initMetal() -> (MTLDevice, MTLCommandQueue, MTLLibrary, MTLCommandBuffer,
        MTLComputeCommandEncoder)
    {
        // Get access to iPhone or iPad GPU
        let device = MTLCreateSystemDefaultDevice()
            
        // Queue to handle an ordered list of command buffers
        let commandQueue = device!.newCommandQueue()
            
        // Access to Metal functions that are stored in Shaders.metal file, e.g. sigmoid()
        let defaultLibrary = device!.newDefaultLibrary()
            
        // Buffer for storing encoded commands that are sent to GPU
        let commandBuffer = commandQueue.commandBuffer()
            
        // Encoder for GPU commands
        let computeCommandEncoder = commandBuffer.computeCommandEncoder()
        computeCommandEncoder.endEncoding();
            
        return (device!, commandQueue, defaultLibrary!, commandBuffer, computeCommandEncoder)
    }


    override func didReceiveMemoryWarning()
    {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

