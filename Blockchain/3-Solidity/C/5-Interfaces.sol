pragma solidity ^0.5.0;

// Interface definition
interface Calculator {
    function getResult() external view returns (uint);
}

// Implementing contract
contract Test is Calculator {
    constructor() public {}

    function getResult() external view returns (uint) {
        uint a = 1;
        uint b = 2;
        uint result = a + b;
        return result;
    }
}
