//  Write a Solidity program that demonstrates various types of functions including regular
//  functions, view functions, pure functions, and the fallback function.


// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract Test {
    uint256 public stateVariable;

    // Regular function (modifies state)
    function setStateVariable(uint256 _value) public {
        stateVariable = _value;
    }

    // View function (reads state, doesn't modify it)
    function getStateVariable() public view returns (uint256) {
        return stateVariable;
    }

    // Pure function (does not read or modify state)
    function returnExample() public pure returns (
        uint256 sum,
        uint256 prod,
        uint256 diff,
        string memory message
    ) {
        uint256 num1 = 10;
        uint256 num2 = 16;
        sum = num1 + num2;
        prod = num1 * num2;
        diff = num2 - num1;
        message = "Multiple return values";
    }

    // Fallback function (called when no other function matches)
    fallback() external payable {
        stateVariable = 999;
    }

    // Receive function (to accept plain Ether transfers)
    receive() external payable {}
}
