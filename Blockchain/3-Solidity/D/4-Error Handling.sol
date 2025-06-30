// SPDX-License-Identifier: MIT
pragma solidity ^0.5.17;

contract ErrorDemo {
    function getSum(uint256 a, uint256 b) public pure returns (uint256) {
        uint256 sum = a + b;

        // Assert that the result is less than 255 (this is a safety condition)
        assert(sum < 255);

        return sum;
    }
}
